"""Benchmark vLLM throughput: baseline vs MTP speculative decoding.

Two invocation modes:

  Single-config:
    python run_benchmark.py --mode baseline --target-path <target_view>
    python run_benchmark.py --mode mtp --target-path <target_view> --draft-path <draft_view>

  Sweep (runs baseline + MTP across multiple (num_prompts, max_tokens) configs,
  spawning a fresh subprocess per run to keep vLLM/CUDA state isolated):
    python run_benchmark.py --sweep --target-path <target_view> --draft-path <draft_view>

Notes on measurement quality:
  * Each LLM() instance does its own internal warmup at init. With spec decoding,
    additional shape buckets get captured lazily on the first generate() call —
    the --warmup flag (default on) runs one untimed generate() with *distinct*
    dummy prompts to flush that deferred work before the timed run.
  * The dummy warmup prompts share zero prefix with the real test prompts, so
    vLLM's prefix cache cannot bleed warmup state into the timed run.
  * Sweep mode spawns a clean subprocess per (mode, config) so vLLM state never
    leaks across runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time


SWEEP_CONFIGS = [
    # (num_prompts, max_tokens) — small online, typical online, heavy/batched
    (8, 256),
    (32, 256),
    (64, 512),
]

PROMPT_POOL = [
    "What is the capital of Japan?",
    "Explain photosynthesis in one paragraph:",
    "The quick brown fox jumps over",
    "List five differences between TCP and UDP:",
    "Write a Python function that reverses a string:",
    "Once upon a time, in a small village by the sea,",
    "Translate to French: 'The weather is beautiful today.'",
    "What are the main causes of climate change?",
    "Summarize the plot of Romeo and Juliet:",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The mitochondria is the powerhouse of the",
    "Compose a haiku about autumn leaves:",
    "Compare and contrast machine learning and deep learning:",
    "In the year 2050, scientists discovered",
    "The first president of the United States was",
    "Step by step, solve this equation: 3x + 7 = 22",
]

WARMUP_POOL = [
    "zzzz random dummy text alpha bravo charlie delta echo foxtrot",
    "qqqq lorem ipsum dolor sit amet consectetur adipiscing elit",
    "wwww unrelated filler content for cuda graph capture purposes only",
    "xxxx synthetic warmup prompt to exercise decode shape bucket here",
    "yyyy this string is solely for triggering torch compile passes",
    "kkkk arbitrary unique tokens distinct from real test prompts",
    "vvvv prefix cache poison prevention dummy line for benchmark",
    "jjjj completely different vocabulary than the actual evaluation",
]


def tile(pool, n):
    return [pool[i % len(pool)] for i in range(n)]


def collect_spec_decode_metrics() -> dict:
    """Pull speculative-decoding counters from vLLM's Prometheus registry."""
    try:
        from prometheus_client import REGISTRY
        samples = {}
        for fam in REGISTRY.collect():
            for s in fam.samples:
                if "spec_decode" in s.name:
                    samples[s.name] = s.value
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


def run_single(args) -> dict:
    """Run one (mode, num_prompts, max_tokens) configuration."""
    from vllm import LLM, SamplingParams

    test_prompts = tile(PROMPT_POOL, args.num_prompts)
    warmup_prompts = tile(WARMUP_POOL, args.num_prompts)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    llm_kwargs = dict(
        model=args.target_path,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
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

    if args.warmup:
        print(f"Warming up {label} with dummy prompts...")
        warmup_start = time.perf_counter()
        llm.generate(warmup_prompts, sampling_params)
        print(f"  Warmup elapsed: {time.perf_counter() - warmup_start:.3f}s")
    else:
        print(f"Skipping warmup for {label} (cold-start measurement).")

    print(f"Running {label} benchmark...")
    start = time.perf_counter()
    outputs = llm.generate(test_prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed

    result = {
        "label": label,
        "mode": args.mode,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "enforce_eager": args.enforce_eager,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "throughput": throughput,
        "texts": [o.outputs[0].text for o in outputs],
    }
    if args.mode == "mtp":
        result.update(collect_spec_decode_metrics())

    print(f"\n{'='*50}")
    print(f"{label} Results:")
    print(f"  num_prompts/max_tokens: {args.num_prompts}/{args.max_tokens}")
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

    return result


def run_sweep(args) -> int:
    """Iterate SWEEP_CONFIGS × {baseline, mtp}, spawning a clean subprocess per run."""
    outdir = args.output_dir or os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)

    results = []
    for nprompts, maxtokens in SWEEP_CONFIGS:
        for mode in ("baseline", "mtp"):
            tag = f"np{nprompts}_mt{maxtokens}"
            if not args.warmup:
                tag += "_nowarmup"
            outfile = os.path.join(outdir, f"sweep_{mode}_{tag}.json")

            cmd = [
                sys.executable, __file__,
                "--mode", mode,
                "--target-path", args.target_path,
                "--num-prompts", str(nprompts),
                "--max-tokens", str(maxtokens),
                f"--{'enforce-eager' if args.enforce_eager else 'no-enforce-eager'}",
                f"--{'warmup' if args.warmup else 'no-warmup'}",
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--num-speculative-tokens", str(args.num_speculative_tokens),
                "--output-file", outfile,
            ]
            if mode == "mtp":
                if not args.draft_path:
                    print("ERROR: --draft-path required for sweep mode (includes MTP runs).", file=sys.stderr)
                    return 1
                cmd += ["--draft-path", args.draft_path]

            print(f"\n=== {mode} | num_prompts={nprompts} max_tokens={maxtokens} ===")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"ERROR: subprocess failed for {mode} {tag} (rc={rc})", file=sys.stderr)
                return rc
            with open(outfile) as f:
                results.append(json.load(f))

    print("\n=== SUMMARY ===")
    by_cfg = {}
    for r in results:
        by_cfg.setdefault((r["num_prompts"], r["max_tokens"]), []).append(r)
    for (nprompts, maxtokens), rs in by_cfg.items():
        print(f"\n--- num_prompts={nprompts} max_tokens={maxtokens} ---")
        for r in rs:
            ar = r.get("acceptance_rate")
            ar_s = f"  accept={ar:.1%}" if ar is not None else ""
            print(f"  {r['label']:8s}  throughput={r['throughput']:7.2f} tok/s  "
                  f"total={r['total_tokens']:6d}  elapsed={r['elapsed']:.2f}s{ar_s}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "mtp"], default=None,
                   help="Single-config mode; ignored when --sweep is set.")
    p.add_argument("--sweep", action="store_true",
                   help="Run SWEEP_CONFIGS × {baseline, mtp}, one subprocess per run.")
    p.add_argument("--target-path", type=str, required=True)
    p.add_argument("--draft-path", type=str, default=None)
    p.add_argument("--num-prompts", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--num-speculative-tokens", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--output-file", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for sweep-mode JSON outputs (default: ./results/).")
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    if args.sweep:
        return run_sweep(args)
    if args.mode is None:
        p.error("--mode is required when not using --sweep")
    run_single(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
