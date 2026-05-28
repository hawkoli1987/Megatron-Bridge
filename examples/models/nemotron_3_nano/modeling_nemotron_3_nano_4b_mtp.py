"""HuggingFace MTP retrofit class for NVIDIA-Nemotron-3-Nano-4B-BF16.

This module is loaded via `trust_remote_code` from a reference HF directory
containing both `configuration_nemotron_h.py` and `modeling_nemotron_h.py`
(the upstream NemotronH classes). It defines `NemotronHMTPConfig` and
`NemotronHMTPForCausalLM` at module top-level by directly subclassing the
upstream classes — no AutoConfig recursion.

Layout matches what NemotronHBridge expects: HF MTP is a flat ModuleList
`mtp.layers[outer*L + inner]`, with wrapper params (enorm/hnorm/eh_proj/
final_layernorm) attached at first/last inner positions of each outer group.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .configuration_nemotron_h import NemotronHConfig
from .modeling_nemotron_h import (
    NemotronHBlock,
    NemotronHForCausalLM,
    NemotronHRMSNorm,
)


_TYPE_FROM_CHAR = {"*": "attention", "-": "mlp", "M": "mamba"}


class NemotronHMTPConfig(NemotronHConfig):
    model_type = "nemotron_h_mtp"

    def __init__(
        self,
        mtp_num_layers: int = 4,
        mtp_hybrid_override_pattern: str = "*-",
        **kwargs,
    ):
        # NemotronHConfig.__init__ validates `hybrid_override_pattern`. The MTP
        # field uses the same name suffix; consume it here so the parent doesn't
        # see it.
        super().__init__(**kwargs)
        self.mtp_num_layers = mtp_num_layers
        # Named to match what NemotronHBridge looks for when computing
        # _mtp_layers_per_block during mapping_registry().
        self.mtp_hybrid_override_pattern = mtp_hybrid_override_pattern



def _shadow_block_config(config, block_type: str):
    """Build a 1-element config so NemotronHBlock(layer_idx=0) is the target type.

    NemotronHConfig.layers_block_type is a property derived from
    hybrid_override_pattern (length must equal num_hidden_layers). Set both.
    """
    cfg = copy.copy(config)
    cfg.hybrid_override_pattern = {
        "attention": "*",
        "mlp": "-",
        "mamba": "M",
    }[block_type]
    cfg.num_hidden_layers = 1
    return cfg


class NemotronHMTPForCausalLM(NemotronHForCausalLM):
    config_class = NemotronHMTPConfig

    def __init__(self, config: NemotronHMTPConfig):
        super().__init__(config)

        outer = getattr(config, "mtp_num_layers", 4)
        per_head = getattr(config, "mtp_hybrid_override_pattern", "*-")
        L = len(per_head)
        h = config.hidden_size
        eps = config.layer_norm_epsilon

        mtp_layers = nn.ModuleList()
        for o in range(outer):
            for i, ch in enumerate(per_head):
                btype = _TYPE_FROM_CHAR[ch]
                blk_cfg = _shadow_block_config(config, btype)
                blk = NemotronHBlock(blk_cfg, layer_idx=0)
                if i == 0:
                    blk.enorm = NemotronHRMSNorm(h, eps=eps)
                    blk.hnorm = NemotronHRMSNorm(h, eps=eps)
                    blk.eh_proj = nn.Linear(h * 2, h, bias=False)
                if i == L - 1:
                    blk.final_layernorm = NemotronHRMSNorm(h, eps=eps)
                mtp_layers.append(blk)

        self.mtp = nn.Module()
        self.mtp.layers = mtp_layers
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        return super().forward(input_ids=input_ids, **kwargs)
