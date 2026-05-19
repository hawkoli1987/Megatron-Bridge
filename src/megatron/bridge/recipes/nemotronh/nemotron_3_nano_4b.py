# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pre-training recipe for NVIDIA-Nemotron-3-Nano-4B-BF16.

Architecture comes verbatim from the HF model card's `config.json` (NemotronH).
This is the *dense* Mamba-Transformer hybrid distilled from Nemotron-Nano-9B-v2
via the Nemotron Elastic framework — NOT to be confused with the existing
`nemotronh_4b_pretrain_config` (a different 4B NemotronH with 52 layers,
hidden 3072, ffn 12288) or `nemotron_3_nano_pretrain_config` (the 30B-A3B MoE).

Mapping HF config.json → MambaModelProvider:
    num_hidden_layers=42        → num_layers=42
    hidden_size=3136            → hidden_size=3136
    intermediate_size=12544     → ffn_hidden_size=12544
    num_attention_heads=40      → num_attention_heads=40
    num_key_value_heads=8       → num_query_groups=8     (GQA)
    head_dim=128                → kv_channels=128
    mamba_num_heads=96          → mamba_num_heads=96
    mamba_head_dim=80           → mamba_head_dim=80
    ssm_state_size=128          → mamba_state_dim=128
    n_groups=8                  → mamba_num_groups=8
    vocab_size=131072           → make_vocab_size_divisible_by=128 (131072=128*1024)
    mlp_hidden_act="relu2"      → activation_func=squared_relu
    initializer_range=0.02      → init_method_std=0.02
    hybrid_override_pattern     → hybrid_layer_pattern   (verbatim — both HF
                                  and MB use `-` for dense FFN. `E` in MB
                                  denotes MoE expert layers, which this
                                  dense model does NOT have; cf. the existing
                                  dense `nemotronh_4b_pretrain_config` which
                                  also uses `M`/`-`/`*` symbols.)

Pattern (length 42 == num_layers): M=Mamba, -=MLP, *=Attention.
"""

import torch
from megatron.core.activations import squared_relu

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer


# `M`=Mamba, `-`=dense MLP (FFN), `*`=Attention. Length must equal num_layers.
# Pattern copied verbatim from HF `hybrid_override_pattern`.
_HYBRID_LAYER_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
assert len(_HYBRID_LAYER_PATTERN) == 42


def nemotron_3_nano_4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for NVIDIA-Nemotron-3-Nano-4B-BF16.

    Dense Mamba-Transformer hybrid (no MoE). Recommended parallelism: TP=1, PP=1
    on a single H100/B200 node (model is ~4B params ≈ 8 GB in BF16).
    """
    cfg = _pretrain_common()

    # Model Configuration (dense Mamba-Transformer hybrid)
    cfg.model = MambaModelProvider(
        # Architecture — from HF config.json
        hybrid_layer_pattern=_HYBRID_LAYER_PATTERN,
        num_layers=42,
        hidden_size=3136,
        ffn_hidden_size=12544,
        num_attention_heads=40,
        num_query_groups=8,
        kv_channels=128,
        seq_length=4096,
        # Mamba-2 specific
        mamba_num_heads=96,
        mamba_head_dim=80,
        mamba_state_dim=128,
        mamba_num_groups=8,
        # NemotronH base (mirrors nemotron_3_nano.py 30B-A3B settings; these are
        # NemotronH-family invariants, not MoE-specific)
        make_vocab_size_divisible_by=128,
        activation_func=squared_relu,
        masked_softmax_fusion=True,
        apply_query_key_layer_scaling=False,
        persist_layer_norm=True,
        attention_softmax_in_fp32=False,
        first_last_layers_bf16=True,
        is_hybrid_model=True,
        # Parallelism — dense 4B fits per-GPU; default to TP=1/PP=1.
        # No EP / expert_* since this is not MoE.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
    )

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

    # Dataset Configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data
    cfg.dataset.num_workers = 8
    # Note: 30B-A3B recipe sets mmap_bin_files=False, but the IndexedDataset
    # class in this Megatron build only supports the mmap path. Leave the
    # default (True) here for compatibility with standard .bin/.idx files.

    # Parallelism — explicit None for layout (no manual PP layout)
    cfg.model.pipeline_model_parallel_layout = None

    # Training Configuration — placeholders; launcher overrides train_iters/GBS
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Transformer Engine
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph — disabled (consistent with both reference recipes' defaults)
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory Saving — defaults off (4B fits without recompute)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer Precision Settings (matches NemotronH family)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Optimizer hyperparameters — Nemotron-family defaults (smaller-model variant).
    # The 30B-A3B uses lr=1.6e-3 with warmup_iters=333 for its full 39735-iter run.
    # For a generic 4B pretrain we keep a conservative cosine-friendly LR; the
    # launcher's WSD scheduler + warmup_ratio takes precedence anyway.
    cfg.optimizer.lr = 4e-4
    cfg.optimizer.min_lr = 4e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.lr_warmup_iters = 200

    # Checkpoint Configuration
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # Init / fusion knobs — match nemotron_3_nano (30B-A3B) NemotronH defaults
    cfg.model.init_method_std = 0.02  # HF initializer_range=0.02
    cfg.model.apply_rope_fusion = False
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg


__all__ = [
    "nemotron_3_nano_4b_pretrain_config",
]
