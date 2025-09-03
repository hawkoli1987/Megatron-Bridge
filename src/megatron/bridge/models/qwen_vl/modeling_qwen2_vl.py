# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core import parallel_state, tensor_parallel
from megatron.core import parallel_state as ps
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference_params import InferenceParams
from megatron.core.jit import jit_fuser
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupsCollection
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params, get_batch_on_this_cp_rank, make_viewless_tensor
from torch import Tensor, nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.vlm.neva.model.base import restore_model_weights
from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class VisionRotaryEmbedding(torch.nn.Module):
    """Generates rotary positional frequencies for vision tokens.

    Computes inverse-frequency based sinusoidal components used by rotary
    position embeddings for attention heads in vision transformers.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @jit_fuser
    def forward(self, seqlen: int) -> torch.Tensor:
        """Return rotary frequencies for a given sequence length.

        Args:
            seqlen: Number of tokens (sequence length).

        Returns:
            Tensor of shape [seqlen, dim/2] with rotary frequencies.
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_VLVisionModel(VisionModule):
    """Qwen2-VL vision model."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        add_class_token: bool = False,
        class_token_len: int = 1,
        patch_dim: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        spatial_patch_size: int = 14,
        img_h: int = 336,
        img_w: int = 336,
    ) -> None:

        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.embed_dim
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.merge_hidden_size = self.visual_hidden_size * (spatial_merge_size**2)
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = 3

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        kernel_size = [temporal_patch_size, patch_dim, patch_dim]
        self.conv1 = torch.nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.visual_hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        head_dim = transformer_config.embed_dim // transformer_config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = torch.nn.Parameter(torch.randn(1, self.class_token_len, self.visual_hidden_size))

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        # pylint: disable=C0115,C0116
        """Compute rotary position embeddings for a grid of [T, H, W] tiles.

        Args:
            grid_thw: Tensor of shape [num_images, 3] containing temporal (T),
                height (H), and width (W) grid sizes after patching/merging.

        Returns:
            Tensor of shape [sum(T*H*W), head_dim] with per-token rotary embeddings.
        """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_packed_seq_params(self, grid_thw):
        # pylint: disable=C0115,C0116
        """Build packed sequence parameters for window attention.

        Args:
            grid_thw: Tensor of shape [N, 3] with [T, H, W] per sample.

        Returns:
            PackedSeqParams with cumulative sequence lengths and max sequence length.
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        cu_seqlens = cu_seqlens.squeeze()  # remove batch size dimension (mbs=1)
        # remove -1 "paddings" added in collate_fn
        # cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

        # pre-compute max_seqlens in dataset class for perf
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # these args are passed eventually into TEDotProductAttention.forward()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
        )

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for Qwen2 vision encoder.

        Processes image patches with a strided 3D convolution, applies rotary
        position embeddings, and passes tokens through the transformer.

        Args:
            x: Input tensor compatible with view into
                [-1, in_channels, temporal_patch_size, patch_dim, patch_dim].
            grid_thw: Tensor of shape [N, 3] with temporal, height, width grid sizes
                per sample after patching/merging.
            attention_mask: Optional boolean attention mask.

        Returns:
            Tensor of shape [num_tokens, merge_hidden_size] after the transformer.
        """
        # pylint: disable=C0301
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_dim, self.patch_dim)
        x = self.conv1(x).view(-1, self.visual_hidden_size)  # [seqlen, hidden_size]
        # add batch dim
        x = x.unsqueeze(1)  # [seqlen, 1, hidden_size], THD format, bs=1
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L158
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L164
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]

        packed_seq_params = self.get_packed_seq_params(grid_thw)
        x = self.decoder(x, attention_mask, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params)

        x = x.squeeze(1).view(-1, self.merge_hidden_size)
        return x


def qwen2vl_data_step(dataloader_iter, model_version) -> Dict[str, torch.Tensor]:
    """Qwen2VL Data Step"""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842
    batch = next(dataloader_iter)
    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    if model_version == "qwen2-vl":
        required_keys.update(("input_ids", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"))
    elif model_version == "qwen25-vl":
        required_keys.update(
            (
                "input_ids",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "second_per_grid_ts",
            )
        )
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("position_ids",))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(
            (
                "labels",
                "loss_mask",
            )
        )

    _batch = {
        key: val.cuda(non_blocking=True) if key in required_keys and val is not None else None
        for key, val in _batch.items()
    }
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch)
    return output


def qwen2vl_forward_step(model, batch) -> torch.Tensor:
    # pylint: disable=C0115,C0116
    forward_args = {
        "input_ids": batch["input_ids"],
        "pixel_values": batch.get("pixel_values", None),
        "image_grid_thw": batch.get("image_grid_thw", None),
        "pixel_values_videos": batch.get("pixel_values_videos", None),
        "video_grid_thw": batch.get("video_grid_thw", None),
        "second_per_grid_ts": batch.get("second_per_grid_ts", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }
    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)
    return model(**forward_args)


def set_input_tensor(self, tensor):
    # pylint: disable=C0115,C0116
    pass


class MCoreQwen2VLModel(MCoreLLaVAModel):
    """Qwen2VL Model Base Model Class"""

    def __init__(
        self,
        config,
        tokenizer: Optional = None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: Optional[int] = None,
    ) -> None:
        # pylint: disable=C0115,C0116
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.language_transformer_config
        vision_transformer_config = config.vision_transformer_config
        vision_projection_config = config.vision_projection_config
        self.model_version = vision_transformer_config.model_version
        assert self.model_version is not None

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size

        self.share_embeddings_and_output_weights = False

        if self.add_decoder:
            self.language_model = language_transformer_config.configure_model(
                tokenizer=tokenizer,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )

            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            self._language_max_sequence_length = self.language_model.max_sequence_length
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1
            restore_model_weights(self.language_model, config.language_model_from_pretrained)
            logging.info(f"Restored language model weights from {config.language_model_from_pretrained}")
        else:
            if config.language_model_from_pretrained is not None:
                dist_checkpointing.load(
                    sharded_state_dict=dict(state_dict={}),
                    checkpoint_dir=config.language_model_from_pretrained,
                    validate_access_integrity=False,
                )

        if self.add_encoder:
            self.vision_model = vision_transformer_config.configure_model()
            self.vision_projection = vision_projection_config.configure_model()
            self._drop_vision_class_token = drop_vision_class_token
            restore_model_weights(self.vision_model, config.vision_model_from_pretrained)
            logging.info(f"Restored vision model weights from {config.vision_model_from_pretrained}")

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder
        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.

        from nemo.collections.vlm.vision.base import get_image_sequence_length
        self._img_seq_len = get_image_sequence_length(
            img_h=vision_transformer_config.img_h,
            img_w=vision_transformer_config.img_w,
            patch_dim=vision_transformer_config.patch_dim,
            add_class_token=not drop_vision_class_token,
            class_token_len=vision_transformer_config.class_token_len,
        )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Qwen2-VL and Qwen25-VL has differnt type:
            Qwen2-VL Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
            Qwen25-VL Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each
                    second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens"
                    are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens
                    per second. So each second of the video will be represented with 25 separate time points. It
                    essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as
                    tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each
                    temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = 2  # self.config.vision_config.spatial_merge_size
        image_token_id = IMAGE_TOKEN_INDEX
        video_token_id = VIDEO_TOKEN_INDEX
        vision_start_token_id = 151652  # self.config.vision_start_token_id
        tokens_per_second = 2
        if second_per_grid_ts is not None:
            second_per_grid_ts = second_per_grid_ts.cpu()

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids.clone()
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids_item in enumerate(total_input_ids):
                _input_ids = input_ids_item[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = _input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = _input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if self.model_version == "qwen25-vl":
                            if second_per_grid_ts is not None:
                                second_per_grid_t = second_per_grid_ts[video_index]
                            else:
                                second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    if self.model_version == "qwen2-vl":
                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    elif self.model_version == "qwen25-vl":
                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        runtime_gather_output: Optional[bool] = None,
        second_per_grid_ts: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            input_ids (torch.Tensor): input text ids [batch, decoder_seq_len].
            attention_mask (torch.Tensor): Attention mask for the language model [batch, 1, combined_seq_len,
            combined_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, decoder_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, decoder_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            pixel_values (torch.Tensor): input image of shape [images_total_patches,
            num_channels * temporal_size * patch_size * patch_size].
            pixel_values_videos (torch.Tensor): input video of shape [videos_total_patches,
            num_channels * temporal_size * patch_size * patch_size].
            image_grid_thw (torch.Tensor): The temporal, height and width of feature shape of each image.
            Shape [num_images, 3].
            video_grid_thw (torch.Tensor): The temporal, height and width of feature shape of each video.
            Shape [num_videos, 3].
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )

        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None

        image_embeddings = None
        if use_inference_kv_cache:
            # If running inference, we can skip media token computation if they were computed already earlier
            # for this sample.
            image_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            image_embeddings = None
        elif self.add_encoder and has_images:
            pixel_values = pixel_values.to(next(self.vision_model.parameters()).dtype)
            if self.config.freeze_vision_model:
                with torch.no_grad():
                    image_embeddings = self.vision_model(
                        pixel_values, grid_thw=image_grid_thw
                    )  # [bs, img_seq_len, h_vision]
            else:
                image_embeddings = self.vision_model(
                    pixel_values, grid_thw=image_grid_thw
                )  # [bs, img_seq_len, h_vision]
            window_index = self.vision_model.window_index if self.model_version == "qwen25-vl" else None

            if self._drop_vision_class_token:
                class_token_len = getattr(self.vision_model, "class_token_len", 1)
                image_embeddings = image_embeddings[:, class_token_len:, :]
                if self.model_version == "qwen25-vl":
                    window_index = [idx - class_token_len for idx in window_index if idx >= class_token_len]
                else:
                    window_index = None

            image_embeddings = self.vision_projection(image_embeddings)
            if self.model_version == "qwen25-vl":
                reverse_indices = torch.argsort(window_index)
                image_embeddings = image_embeddings[reverse_indices, :]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["media_tokens_count"] = (
                    image_embeddings.shape[1] * image_embeddings.shape[2]
                )
        else:
            image_embeddings = self.encoder_hidden_state

        video_embeddings = None
        if self.add_encoder and has_videos:
            pixel_values_videos = pixel_values_videos.to(next(self.vision_model.parameters()).dtype)
            if self.config.freeze_vision_model:
                with torch.no_grad():
                    video_embeddings = self.vision_model(
                        pixel_values_videos, grid_thw=video_grid_thw
                    )  # [bs, img_seq_len, h_vision]
            else:
                video_embeddings = self.vision_model(
                    pixel_values_videos, grid_thw=video_grid_thw
                )  # [bs, img_seq_len, h_vision]
            video_embeddings = self.vision_projection(video_embeddings)
        if not self.add_decoder:
            return image_embeddings

        # language_embeddings is a container for text, image and video embeddings; to feed to decoder
        language_embeddings = None

        language_seq_len = input_ids.shape[1]
        # chunk if input seq_len > _language_max_sequence_length
        if language_seq_len > self._language_max_sequence_length:
            input_ids = input_ids[:, : self._language_max_sequence_length]
            if position_ids is not None:
                position_ids = position_ids[:, :, : self._language_max_sequence_length]

            if labels is not None and labels.shape[1] > self._language_max_sequence_length:
                labels = labels[:, : self._language_max_sequence_length]
                loss_mask = loss_mask[:, : self._language_max_sequence_length]

        # Pipeline parallel expects fixed input size. Check if we need to pad.
        if self._language_is_pipeline_parallel and language_seq_len < self._language_max_sequence_length:
            padded_seq_len = self._language_max_sequence_length - language_seq_len
            input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq_len))
            if position_ids is not None:
                position_ids = torch.nn.functional.pad(position_ids, (0, padded_seq_len))

        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
            )

        # Create the language_embeddings (if this is the first language model stage).
        if self.pre_process:

            # Note: This adds absolute position embedding but not RoPE.
            # Each image is counted as one position.
            # RoPE is added in language_model forward. Each image embedding is one position.
            if self.sequence_parallel_lm:
                # Pad to nearest multiple of TP world size for embedding.
                tp_world_size = ps.get_tensor_model_parallel_world_size()
                padded_seq_len = (
                    int((input_ids.shape[1] + tp_world_size - 1) // tp_world_size * tp_world_size) - input_ids.shape[1]
                )
                if padded_seq_len != 0:
                    input_ids = torch.nn.functional.pad(input_ids, (0, padded_seq_len))
                    if position_ids is not None:
                        position_ids = torch.nn.functional.pad(position_ids, (0, padded_seq_len))

            input_ids_text = input_ids.clone()
            # MultiModal Token indices are assumed to be values
            input_ids_text[input_ids_text < 0] = 0

            language_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=None
            )  # [decoder_seq_len, b, h_language]

            language_embeddings = language_embeddings.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

        # Preprocess input, labels and loss mask.
        combined_embeddings, final_labels, final_loss_mask, final_attention_mask = self._preprocess_data(
            input_ids,
            loss_mask=loss_mask,
            labels=labels,
            language_embeddings=language_embeddings,
            image_embeddings=image_embeddings,
            video_embeddings=video_embeddings,
            attention_mask=attention_mask,
        )  # [decoder_seq_len, b, h_language], [b, decoder_seq_len], [b, decoder_seq_len]

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=final_attention_mask,
            decoder_input=combined_embeddings,
            labels=final_labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
        )  # output shape: [batch_size, seq length, vocab_size]

        if labels is None or loss_mask is None:
            return output
        else:
            return output, final_loss_mask.contiguous()

    # override _preprocess_data() in megatron-lm/megatron/core/models/multimodal/llava_model.py
    def _preprocess_data(
        self,
        input_ids: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        language_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        video_embeddings: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_inference_kv_cache: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        MCoreQwen2VLModel uses its own version of _preprocess_data instead of MCoreLLaVAModel's (in
        megatron-lm/megatron/core/models/multimodal/llava_model.py)

        This function handles several data preprocess requirements:
            - merge image and/or video embeddings into language embedding
            - padding inputs variables (e.g. labels/loss masks) for pipeline_parallel case
            - truncate inputs variables (e.g. labels/loss masks) if exceeding max seq length

        This function won't shift labels as forward() and _preprocess_data() in MCoreQwen2VLModel
        expect labels from input arguments already handle this shift.

        About merging image/video embeddings: language_embeddings may include num of imgage_token
        placeholders, and this function will put each imgage_token from image_embeddings into
        placeholder within language_embeddings(1:1 mapping), when image_embeddings/video_embeddings
        is available and it's the 1st pipeline_parallel stage
        """

        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed.
        # With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return None, None, None, None

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels, attention_mask

        # img_seq_len = self._img_seq_len
        batch_size, language_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        has_images = image_embeddings is not None
        has_videos = video_embeddings is not None

        #
        # Create the final input embedding (if this is the first language model stage).
        #
        final_embedding = None
        if self.pre_process:
            final_embedding = language_embeddings

            # merge image embeddings into language_embeddings
            if has_images:
                # has images, merge image_embeddings into final_embedding
                n_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
                n_image_features = image_embeddings.shape[0]

                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                        f"features {n_image_features}"
                    )

                image_mask = (
                    (input_ids == IMAGE_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(final_embedding)
                    .to(final_embedding.device)
                )
                image_embeddings = image_embeddings.to(final_embedding.device, final_embedding.dtype)
                final_embedding = final_embedding.masked_scatter(
                    image_mask, image_embeddings
                )  #  [b, seq_len, h_language]

            # merge video embeddings into final_embedding
            if has_videos:
                # has images, merge image_embeddings into final_embedding
                n_video_tokens = (input_ids == VIDEO_TOKEN_INDEX).sum().item()
                n_video_features = video_embeddings.shape[0]

                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                        f"features {n_video_features}"
                    )

                video_mask = (
                    (input_ids == VIDEO_TOKEN_INDEX)
                    .unsqueeze(-1)
                    .expand_as(final_embedding)
                    .to(final_embedding.device)
                )
                video_embeddings = video_embeddings.to(final_embedding.device, final_embedding.dtype)
                final_embedding = final_embedding.masked_scatter(video_mask, video_embeddings)

        #
        # Create the final labels and loss mask (if this is the last language model stage).
        #
        final_labels, final_loss_mask = None, None

        if self.post_process and has_labels:

            # Pipeline parallel expects fixed input size. Check if we need to pad
            if self._language_is_pipeline_parallel and labels.shape[1] < self._language_max_sequence_length:
                max_seq_len = self._language_max_sequence_length
                final_labels = torch.full(
                    (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
                )
                final_loss_mask = torch.full(
                    (batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device
                )
                final_labels[:, : labels.shape[1]] = labels[:, :]
                final_loss_mask[:, : labels.shape[1]] = loss_mask[:, :]
            else:
                final_labels, final_loss_mask = labels, loss_mask

        if final_embedding is not None and final_labels is not None:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            # Truncate if exceeding the language model's max sequence length.
            if final_embedding.shape[1] > self._language_max_sequence_length:
                final_embedding = final_embedding[:, : self._language_max_sequence_length]

            # TODO: check and add self.context_parallel_lm to MCoreQwen2VLModel
            # # Transpose to [s,b,h] if not using CP because CP Sharding expects seq in dim=1
            final_embedding = final_embedding.transpose(1, 0).contiguous()  #  [seq_len, bs, h_language]
            if self.sequence_parallel_lm:
                final_embedding = scatter_to_sequence_parallel_region(final_embedding)
        truncate_labels = final_labels is not None and final_labels.shape[1] > self._language_max_sequence_length
        if truncate_labels:
            final_labels = final_labels[:, : self._language_max_sequence_length]
            final_loss_mask = final_loss_mask[:, : self._language_max_sequence_length]
        return final_embedding, final_labels, final_loss_mask, attention_mask

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for llava'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])


class Qwen2VLModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """Lightning Wrapper for Qwen2VL Model"""

    def __init__(
        self,
        config,
        model_version: str,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        # pylint: disable=C0115,C0116
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self.model_version = model_version
        assert self.model_version in ["qwen2-vl", "qwen25-vl"], "model_version only supports qwen2-vl and qwen25-vl."

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer, vp_stage=vp_stage)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            labels=labels,
            inference_params=inference_params,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        # pylint: disable=C0115,C0116
        return self.config.data_step_fn(dataloader_iter, self.model_version)

    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction


__all__ = [
    "VisionRotaryEmbedding",
    "Qwen2_VLVisionModel", 
    "qwen2vl_data_step",
    "qwen2vl_forward_step",
    "set_input_tensor",
    "MCoreQwen2VLModel",
    "Qwen2VLModel",
]
