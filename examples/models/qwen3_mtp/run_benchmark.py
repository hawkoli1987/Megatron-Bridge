"""Single-mode benchmark: run either baseline or MTP in one invocation.

Usage:
  python run_benchmark.py --mode baseline --target-path /path/to/target_view
  python run_benchmark.py --mode mtp --target-path /path/to/target_view --draft-path /path/to/draft_view

Designed to be called directly from enroot (one invocation per mode)
to avoid subprocess nesting that breaks CUDA in enroot containers.
"""
import json
import sys
import time
import argparse


def collect_spec_decode_metrics() -> dict:
    """Collect speculative decoding metrics from Prometheus registry."""
    try:
        from prometheus_client import REGISTRY
        samples = {}
        for metric_family in REGISTRY.collect():
            for sample in metric_family.samples:
                if "spec_decode" in sample.name:
                    samples[sample.name] = sample.value
        accepted = samples.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        draft = samples.get("vllm:spec_decode_num_draft_tokens_total", 0)
        if draft > 0:
            return {
                "num_accepted_tokens": int(accepted),
                "num_draft_tokens": int(draft),
                "acceptance_rate": accepted / draft,
            }
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "mtp"], required=True)
    parser.add_argument("--target-path", type=str, required=True)
    parser.add_argument("--draft-path", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    test_prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "Write the numbers 1 to 5:",
    ]
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    llm_kwargs = dict(
        model=args.target_path,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.mode == "mtp":
        assert args.draft_path, "--draft-path required for MTP mode"
        llm_kwargs["disable_log_stats"] = False
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "model": args.draft_path,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        label = "MTP"
    else:
        llm_kwargs["disable_log_stats"] = True
        label = "Baseline"

    print(f"Loading {label} model...")
    llm = LLM(**llm_kwargs)

    print(f"Running {label} benchmark...")
    start_time = time.perf_counter()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    result = {
        "label": label,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "throughput": throughput,
        "texts": [o.outputs[0].text for o in outputs],
    }

    if args.mode == "mtp":
        spec_metrics = collect_spec_decode_metrics()
        if spec_metrics:
            result.update(spec_metrics)

    print(f"\n{'='*50}")
    print(f"{label} Results:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.2f} tokens/sec")
    if "acceptance_rate" in result:
        print(f"  Acceptance rate: {result['acceptance_rate']:.1%}")
        print(f"  Accepted/Draft: {result['num_accepted_tokens']}/{result['num_draft_tokens']}")
    print(f"{'='*50}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
