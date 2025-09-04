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

from dataclasses import dataclass, field
from typing import List

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models import (
    Qwen2ModelProvider,
)

from .modeling_qwen2p5_vl import Qwen2p5_VLModel


# =============================================================================
# Qwen 2.5 Model Providers
# =============================================================================


@dataclass
class Qwen2p5VLModelProvider(Qwen2ModelProvider):
    """
    Base model provider for Qwen 2.5 VL Models.
    """

    # Language configuration inherited from Qwen2p5ModelProvider3B
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen2_5_VLVisionConfig = field(default_factory=Qwen2_5_VLVisionConfig)

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen2p5_VLModel:
        return Qwen2p5_VLModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Qwen2p5VLModelProvider3B(Qwen2p5VLModelProvider):
    """
    Config for Qwen 2.5-VL 3B Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
    """

    # Language configuration inherited from Qwen2p5ModelProvider3B
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen2_5_VLVisionConfig = field(
        default_factory=lambda: Qwen2_5_VLVisionConfig(
            depth=32,
            hidden_act="silu",
            hidden_size=1280,
            intermediate_size=3420,
            num_heads=16,
            in_chans=3,
            out_hidden_size=2048,
            patch_size=14,
            spatial_merge_size=2,
            spatial_patch_size=14,
            window_size=112,
            fullatt_block_indexes=[7, 15, 23, 31],
            tokens_per_second=2,
            temporal_patch_size=2,
        )
    )

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656


@dataclass
class Qwen2p5VLModelProvider7B(Qwen2p5VLModelProvider):
    """
    Config for Qwen 2.5-VL 7B Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
    """

    # Language configuration inherited from Qwen2p5ModelProvider7B
    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen2_5_VLVisionConfig = field(
        default_factory=lambda: Qwen2_5_VLVisionConfig(
            depth=32,
            hidden_act="silu",
            hidden_size=1280,
            intermediate_size=3420,
            num_heads=16,
            in_chans=3,
            out_hidden_size=2048,
            patch_size=14,
            spatial_merge_size=2,
            spatial_patch_size=14,
            window_size=112,
            fullatt_block_indexes=[7, 15, 23, 31],
            tokens_per_second=2,
            temporal_patch_size=2,
        )
    )

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
