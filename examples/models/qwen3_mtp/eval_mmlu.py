#!/usr/bin/env python3
"""MMLU evaluation using lm-evaluation-harness (lm_eval) with vLLM backend.

Run inside a vLLM container (e.g. vllm+vllm-openai+v0.13.0) that has both
vLLM and lm_eval pre-installed.  No runtime patches needed.
"""
import sys
import json
import argparse
from pathlib import Path

import lm_eval.models.huggingface  # noqa: F401 – register "hf"
try:
    import lm_eval.models.vllm_causallms  # noqa: F401 – register "vllm"
except ImportError:
    pass
from lm_eval import evaluator


def main():
    parser = argparse.ArgumentParser(description="MMLU Evaluation via lm_eval")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or HF model ID")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--tasks", type=str, default="mmlu",
                        help="Comma-separated lm_eval task names")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "hf"],
                        help="lm_eval model backend (default: vllm)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel size (vllm backend only)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max sequence length (vllm backend only)")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.backend == "vllm":
        model_args = (
            f"pretrained={args.model}"
            f",dtype={args.dtype}"
            f",tensor_parallel_size={args.tp}"
            f",max_model_len={args.max_model_len}"
            f",gpu_memory_utilization=0.7"
        )
        backend = "vllm"
        batch_size = "auto"
    else:
        model_args = (
            f"pretrained={args.model}"
            f",dtype={args.dtype}"
            f",trust_remote_code=True"
        )
        backend = "hf"
        batch_size = args.batch_size

    print(f"Running lm_eval on model: {args.model}")
    print(f"  Backend: {backend}")
    print(f"  Tasks: {tasks}")
    print(f"  Few-shot: {args.num_fewshot}, Batch size: {batch_size}")
    print(f"  model_args: {model_args}")

    results = evaluator.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
    )

    print(f"\n{'='*60}")
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none", task_results.get("acc", "N/A"))
        acc_norm = task_results.get("acc_norm,none",
                                    task_results.get("acc_norm", "N/A"))
        print(f"  {task_name}: acc={acc}, acc_norm={acc_norm}")
    print(f"{'='*60}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "model": args.model,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "results": results.get("results", {}),
            "config": {
                "model_args": model_args,
                "backend": backend,
                "batch_size": batch_size,
            },
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
