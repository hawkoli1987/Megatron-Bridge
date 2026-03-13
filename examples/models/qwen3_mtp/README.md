# Qwen3-4B MTP Retrofit Example

Retrofit Multi-Token Prediction (MTP) onto Qwen3-4B using Megatron-Bridge,
then serve with vLLM self-speculative decoding.

## Overview

This example demonstrates a 5-step pipeline:

1. **Import & Warmup** — Load HF checkpoint into Megatron with randomly-initialized MTP layers; freeze backbone and train MTP only.
2. **Joint Training** — Unfreeze all parameters and train jointly with reduced MTP loss weight.
3. **Export** — Convert Megatron checkpoint back to HuggingFace format with MTP weights.
4. **Evaluation** — Run MMLU to verify model quality is preserved.
5. **Inference** — Serve with vLLM using MTP heads as draft tokens for self-speculative decoding.

## Files

| File | Purpose |
|------|---------|
| `pretrain_qwen3_4b_mtp.py` | Training entry script. Uses `qwen3_4b_pretrain_config` recipe with MTP, supports YAML/CLI overrides. |
| `conf/qwen3_4b_mtp_warmup.yaml` | Phase 1 config overrides: LR 1e-4, MTP loss weight 0.3, 2500 iters. |
| `conf/qwen3_4b_mtp_joint.yaml` | Phase 2 config overrides: LR 5e-5, MTP loss weight 0.1, 5000 iters. |
| `modeling_qwen3_mtp.py` | HuggingFace model definition (`Qwen3MTPForCausalLM`). Placed in exported checkpoints for `trust_remote_code`. |
| `qwen3_mtp_config.py` | vLLM draft model config (`model_type="mtp"`). Used by `draft_view/`. |
| `vllm_qwen3_mtp.py` | vLLM MTP model implementation for speculative decoding. |
| `setup_views.py` | Splits exported HF checkpoint into `target_view/` (backbone) and `draft_view/` (MTP heads). |
| `run_benchmark_single.py` | Throughput benchmark: baseline vs MTP speculative decoding. Reports acceptance rate. |
| `eval_mmlu.py` | MMLU evaluation via `lm_eval`. Includes patches for NeMo 25.11 container compatibility. |
| `enroot_rc.sh` | Minimal RC script for enroot container entry. |

The Megatron-Bridge source changes that enable this pipeline:

- `src/megatron/bridge/models/qwen/qwen3_mtp_bridge.py` — MTP weight bridge
- `src/megatron/bridge/training/setup.py` — HF loading + MTP freezing hooks
- `src/megatron/bridge/training/config.py` — `hf_pretrained_checkpoint` config
- `examples/conversion/convert_checkpoints.py` — `--trust-remote-code` for MTP export

## Prerequisites

- **Training**: NeMo container (`nemo_25.11.sif`) with Singularity
- **Inference**: vLLM container (`vllm+vllm-openai+v0.13.0.sqsh`) with enroot
- **Dataset**: FineWeb-Edu 10BT (tokenized for Megatron)
- **Hardware**: 1x NVIDIA H200 node (8 GPUs) for training; 1x H200 GPU for inference

## Quick Start

### Step 1: Train (inside NeMo container)

```bash
# Phase 1: MTP Warmup (~2500 iters, backbone frozen)
torchrun --nproc_per_node=8 examples/models/qwen3_mtp/pretrain_qwen3_4b_mtp.py \
  --config-file examples/models/qwen3_mtp/conf/qwen3_4b_mtp_warmup.yaml \
  --hf-pretrained-checkpoint Qwen/Qwen3-4B \
  --freeze-non-mtp \
  model.seq_length=4096 \
  dataset.blend='[["path/to/tokenized/data"]]'

# Phase 2: Joint Training (~5000 iters, all params)
torchrun --nproc_per_node=8 examples/models/qwen3_mtp/pretrain_qwen3_4b_mtp.py \
  --config-file examples/models/qwen3_mtp/conf/qwen3_4b_mtp_joint.yaml \
  --pretrained-checkpoint /path/to/warmup/checkpoint \
  model.seq_length=4096 \
  dataset.blend='[["path/to/tokenized/data"]]'
```

### Step 2: Export (inside NeMo container)

```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model /path/to/hf-base-dir \
  --megatron-path /path/to/megatron/checkpoint/iter_NNNNN \
  --hf-path /path/to/exported \
  --trust-remote-code --not-strict
```

### Step 3: Setup Views

```bash
python examples/models/qwen3_mtp/setup_views.py \
  --source-dir /path/to/exported \
  --scripts-dir examples/models/qwen3_mtp
```

### Step 4: Evaluate (inside NeMo container)

```bash
python examples/models/qwen3_mtp/eval_mmlu.py \
  --model /path/to/exported/target_view --backend vllm --tp 1
```

### Step 5: Benchmark (inside vLLM container via enroot)

```bash
# Baseline
python examples/models/qwen3_mtp/run_benchmark_single.py \
  --mode baseline --target-path /path/to/exported/target_view

# MTP speculative decoding
python examples/models/qwen3_mtp/run_benchmark_single.py \
  --mode mtp \
  --target-path /path/to/exported/target_view \
  --draft-path /path/to/exported/draft_view
```

## Results (1x H200 GPU, vLLM 0.13, enforce_eager)

| Metric              | Original Qwen3-4B | Warmup (Phase 1) | Joint (Phase 2) |
| ------------------- | ----------------- | ----------------- | --------------- |
| MMLU                | 70.13%            | 70.11%            | 67.36%          |
| Acceptance Rate     | N/A               | 65.2%             | 82.7%           |
| Baseline Throughput | ~255 tok/s        | 256.06 tok/s      | 256.79 tok/s    |
| MTP Throughput      | N/A               | 299.38 tok/s      | 334.83 tok/s    |
| MTP Speedup         | N/A               | +16.9%            | +30.4%          |

See `results/` for raw benchmark and evaluation JSON outputs.
