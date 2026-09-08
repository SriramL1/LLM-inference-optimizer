"""
Stage 1 baseline benchmark.

Runs the unoptimized reference engine across a sweep of batch sizes and
generation lengths, recording TTFT, decode throughput, and peak memory as
mean ± std over multiple trials per config. Every later optimization
stage re-runs this same shape of benchmark and compares its numbers
against benchmarks/results/stage1_baseline.json.

Usage:
    python benchmarks/run_baseline.py
    python benchmarks/run_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --batch-sizes 1 4 8 --gen-lengths 32 128 --trials 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.baseline_engine import BaselineEngine
from src.engine.profiler import InferenceProfiler, save_results, print_summary_table

DEFAULT_PROMPT = (
    "Explain the difference between prefill and decode in transformer "
    "inference, and why they have different performance characteristics."
)


def build_batch(tokenizer, prompt: str, batch_size: int, device: str):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return ids.repeat(batch_size, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--gen-lengths", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="warmup runs to discard (CUDA context / kernel autotune costs)",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="measured trials per config, reported as mean ± std",
    )
    parser.add_argument("--out", default="benchmarks/results/stage1_baseline.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: no CUDA GPU detected, falling back to CPU. "
            "Results will not be meaningful for GPU optimization work -- "
            "run this on your RTX 4070 machine instead."
        )

    print(f"Loading {args.model} on {device}...")
    loaded = load_model(args.model, device=device)
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device=device)
    profiler = InferenceProfiler(device=device)

    results = []
    for batch_size in args.batch_sizes:
        input_ids = build_batch(loaded.tokenizer, args.prompt, batch_size, device)
        for gen_len in args.gen_lengths:
            for _ in range(args.warmup):
                profiler.profile_generation(engine, input_ids, gen_len, args.model)

            result = profiler.profile_generation_repeated(
                engine, input_ids, gen_len, args.model, n_trials=args.trials
            )
            results.append(result)
            print(
                f"batch={batch_size:<3} gen={gen_len:<4} "
                f"TTFT={result.ttft_ms_mean:.1f}±{result.ttft_ms_std:.1f}ms  "
                f"decode={result.decode_tokens_per_sec_mean:.1f}±{result.decode_tokens_per_sec_std:.1f} tok/s  "
                f"peak_mem={result.peak_memory_mb_mean:.0f}MB  (n={args.trials})"
            )

    print("\n=== Stage 1 Baseline Summary ===")
    print_summary_table(results)
    save_results(results, args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()