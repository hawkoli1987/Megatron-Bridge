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
| `run_benchmark.py` | Throughput benchmark: baseline vs MTP speculative decoding. Reports acceptance rate. |
| `eval_mmlu.py` | MMLU evaluation via `lm_eval` with vLLM backend. |
| `enroot_rc.sh` | Minimal RC script for enroot container entry. |

The Megatron-Bridge source changes that enable this pipeline:

- `src/megatron/bridge/models/qwen/qwen3_mtp_bridge.py` — MTP weight bridge (needed for **export** only; during import, the standard `Qwen3Bridge` handles backbone weights while MTP layers are randomly initialized)
- `src/megatron/bridge/training/setup.py` — HF loading + MTP freezing hooks
- `src/megatron/bridge/training/config.py` — `hf_pretrained_checkpoint` config
- `examples/conversion/convert_checkpoints.py` — `--trust-remote-code` for MTP export

## Prerequisites

- **Training & Export**: NeMo container (`nemo_25.11.sif`) with Singularity
- **Evaluation & Inference**: vLLM container (`vllm+vllm-openai+v0.13.0.sqsh`) with enroot; `pip install lm_eval` inside the container for MMLU evaluation
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
  --data-path /path/to/tokenized/data_text_document

# Phase 2: Joint Training (~5000 iters, all params)
torchrun --nproc_per_node=8 examples/models/qwen3_mtp/pretrain_qwen3_4b_mtp.py \
  --config-file examples/models/qwen3_mtp/conf/qwen3_4b_mtp_joint.yaml \
  --pretrained-checkpoint /path/to/warmup/checkpoint \
  --data-path /path/to/tokenized/data_text_document
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

This creates two HF-compatible checkpoint directories from the exported MTP checkpoint:
- **`target_view/`** — backbone-only weights (MTP parameters removed), used as the verifier in speculative decoding.
- **`draft_view/`** — MTP head weights with `modeling_qwen3_mtp.py`, `qwen3_mtp_config.py`, and `vllm_qwen3_mtp.py` copied in, plus `config.json` updated with `"auto_map": {"AutoModelForCausalLM": "modeling_qwen3_mtp.Qwen3MTPForCausalLM"}` so vLLM can load it via `trust_remote_code=True`.

### Steps 4–5: Evaluate & Benchmark (inside vLLM container via enroot)

Both evaluation and inference benchmarks run inside the vLLM container.
Use `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (not `all`) on headless
compute nodes. Do **not** set `VLLM_WORKER_MULTIPROC_METHOD=spawn` — the
default fork mode is required for CUDA to work in vLLM's subprocess.

```bash
export EXPORTED_HF=/path/to/exported
export SHARED_FS=/path/to/shared/filesystem

# MMLU evaluation
enroot start --root --rw \
    --rc examples/models/qwen3_mtp/enroot_rc.sh \
    --mount="${HOME}:${HOME}" \
    --mount="/scratch_aisg:/scratch_aisg" \
    --mount="${SHARED_FS}:${SHARED_FS}" \
    --mount="/tmp:/tmp" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_HOME=${SHARED_FS}/cache/huggingface \
    ${CONTAINER_NAME} \
    python3 eval_mmlu.py \
    --model ${EXPORTED_HF}/target_view --backend vllm --tp 1 \
    --output-file /tmp/mmlu_results.json

# Throughput benchmark — baseline
enroot start --root --rw \
    --rc examples/models/qwen3_mtp/enroot_rc.sh \
    --mount="${HOME}:${HOME}" \
    --mount="/scratch_aisg:/scratch_aisg" \
    --mount="${SHARED_FS}:${SHARED_FS}" \
    --mount="/tmp:/tmp" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_HOME=${SHARED_FS}/cache/huggingface \
    --env PYTHONPATH=${EXPORTED_HF}/draft_view \
    ${CONTAINER_NAME} \
    python3 run_benchmark.py --mode baseline \
    --target-path ${EXPORTED_HF}/target_view \
    --output-file /tmp/benchmark_baseline.json

# Throughput benchmark — MTP speculative decoding
# Same as above but: --mode mtp --draft-path ${EXPORTED_HF}/draft_view
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

## Troubleshooting

### Enroot: `stat failed: /dev/nvidia-modeset: no such file or directory`

Headless compute nodes lack the `nvidia-modeset` device. Use
`NVIDIA_DRIVER_CAPABILITIES=compute,utility` instead of `all` in enroot env.

### Enroot: `CUDA driver initialization failed` in EngineCore subprocess

vLLM 0.13's multiprocess engine spawns a subprocess that cannot inherit the
CUDA context inside enroot. Do **not** set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
The default `fork` mode correctly inherits the parent's CUDA context and
achieves full throughput.
