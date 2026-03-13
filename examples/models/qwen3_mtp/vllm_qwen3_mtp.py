import torch
import torch.nn as nn
from typing import Iterable, Dict
from collections import defaultdict

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

import qwen3_mtp_config

try:
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
except ImportError:
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer as Qwen3DecoderLayer


class Qwen3MTPLayer(nn.Module):
    """Single MTP layer with normalization, projection, and transformer."""

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.layer_idx = layer_idx

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.eh_proj"
        )

        self.transformer_layer = Qwen3DecoderLayer(
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.transformer_layer"
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds,
                spec_step_index=0):
        norm_e = self.enorm(inputs_embeds)
        norm_h = self.hnorm(previous_hidden_states)
        combined = torch.cat([norm_e, norm_h], dim=-1)
        hidden_states, _ = self.eh_proj(combined)

        hidden_states, residual = self.transformer_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=None
        )

        if residual is not None:
            hidden_states = hidden_states + residual

        return self.final_layernorm(hidden_states)

class Qwen3MTPBackbone(nn.Module):
    """MTP backbone with embedding and MTP layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens"
        )

        self.num_mtp_layers = getattr(config, "mtp_num_layers", 1)
        self.layers = nn.ModuleList([
            Qwen3MTPLayer(vllm_config, layer_idx=i, prefix=f"{prefix}.layers.{i}")
            for i in range(self.num_mtp_layers)
        ])

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, spec_step_idx=0, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        layer_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[layer_idx](
            input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_idx
        )

class Qwen3MTPModel(nn.Module):
    """Main MTP model for vLLM speculative decoding."""

    @classmethod
    def is_backend_compatible(cls):
        return True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.model = Qwen3MTPBackbone(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, hidden_states,
                intermediate_tensors=None, inputs_embeds=None, spec_step_idx=0, **kwargs):
        return self.model(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx
        )

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with Q/K/V and gate/up merging."""
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        qkv_weights: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        gate_up_weights: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)

        weights_list = list(weights)

        for name, loaded_weight in weights_list:
            # Skip non-MTP weights
            if "mtp" not in name and "embed_tokens" not in name and "lm_head" not in name:
                continue

            # Embeddings and lm_head are shared with target model
            if "embed_tokens" in name:
                vllm_name = "model.embed_tokens.weight"
                if vllm_name in params_dict:
                    param = params_dict[vllm_name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                    loaded_params.add(vllm_name)
                continue

            if "lm_head" in name:
                vllm_name = "lm_head.weight"
                if vllm_name in params_dict:
                    param = params_dict[vllm_name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, loaded_weight)
                    loaded_params.add(vllm_name)
                continue

            # Map layers
            if name.startswith("mtp."):
                vllm_name = "model.layers." + name[4:]
            else:
                vllm_name = name

            # Collect Q/K/V for merging into qkv_proj
            if ".self_attn.q_proj." in vllm_name:
                layer_idx = int(vllm_name.split(".")[2])
                qkv_weights[layer_idx]["q"] = loaded_weight
                continue
            elif ".self_attn.k_proj." in vllm_name:
                layer_idx = int(vllm_name.split(".")[2])
                qkv_weights[layer_idx]["k"] = loaded_weight
                continue
            elif ".self_attn.v_proj." in vllm_name:
                layer_idx = int(vllm_name.split(".")[2])
                qkv_weights[layer_idx]["v"] = loaded_weight
                continue

            # Collect gate/up for merging into gate_up_proj
            if ".mlp.gate_proj." in vllm_name:
                layer_idx = int(vllm_name.split(".")[2])
                gate_up_weights[layer_idx]["gate"] = loaded_weight
                continue
            elif ".mlp.up_proj." in vllm_name:
                layer_idx = int(vllm_name.split(".")[2])
                gate_up_weights[layer_idx]["up"] = loaded_weight
                continue

            # Load other weights
            if vllm_name in params_dict:
                param = params_dict[vllm_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_params.add(vllm_name)

        # Merge and load QKV weights
        for layer_idx, qkv in qkv_weights.items():
            if all(k in qkv for k in ["q", "k", "v"]):
                merged = torch.cat([qkv["q"], qkv["k"], qkv["v"]], dim=0)
                vllm_name = f"model.layers.{layer_idx}.transformer_layer.self_attn.qkv_proj.weight"
                if vllm_name in params_dict:
                    param = params_dict[vllm_name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, merged)
                    loaded_params.add(vllm_name)

        # Merge and load gate_up weights
        for layer_idx, gu in gate_up_weights.items():
            if all(k in gu for k in ["gate", "up"]):
                merged = torch.cat([gu["gate"], gu["up"]], dim=0)
                vllm_name = f"model.layers.{layer_idx}.transformer_layer.mlp.gate_up_proj.weight"
                if vllm_name in params_dict:
                    param = params_dict[vllm_name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, merged)
                    loaded_params.add(vllm_name)

        # Report loading status
        missing = [p for p in params_dict if p not in loaded_params]
        if missing:
            print(f"[MTP] Warning: missing parameters: {missing}")

        return loaded_params


def register_model():
    """Register Qwen3MTPModel with vLLM."""
    try:
        from vllm.model_executor.models import ModelRegistry
        ModelRegistry.register_model("Qwen3MTPModel", Qwen3MTPModel)
    except ValueError:
        pass


register_model()