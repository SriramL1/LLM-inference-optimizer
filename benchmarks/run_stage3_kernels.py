"""
Stage 3 isolated kernel microbenchmark.

Compares naive (unfused, multi-kernel-launch) PyTorch vs. the Triton
fused kernel, for both RMSNorm and silu-and-mul, across a sweep of row
counts (batch * seq_len flattened -- the axis that matters here, since
these are per-token elementwise/reduction ops, not attention).

Expectation-setting: these are cheap, memory-bound ops -- don't expect
attention-sized speedups. The win is fewer kernel launches and less
memory traffic, which shows up more clearly at smaller row counts where
launch overhead is a bigger fraction of total time, and shrinks
(proportionally) as row counts grow and the ops become more bandwidth-
saturated either way.

Usage:
    python benchmarks/run_stage3_kernels.py
    python benchmarks/run_stage3_kernels.py --rows 64 512 4096 32768
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.kernels.fused_rmsnorm_triton import fused_rmsnorm, naive_rmsnorm_reference
from src.kernels.fused_swiglu_triton import fused_silu_mul, naive_silu_mul_reference


def _time_cuda(fn, n_trials: int, warmup: int = 5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.mean(times_ms), (statistics.stdev(times_ms) if n_trials > 1 else 0.0)


def benchmark_rmsnorm(rows, hidden_size, n_trials, device="cuda", dtype=torch.float16):
    torch.manual_seed(0)
    x = torch.randn(rows, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    naive_mean, naive_std = _time_cuda(lambda: naive_rmsnorm_reference(x, weight), n_trials)
    triton_mean, triton_std = _time_cuda(lambda: fused_rmsnorm(x, weight), n_trials)
    return naive_mean, naive_std, triton_mean, triton_std


def benchmark_swiglu(rows, intermediate_size, n_trials, device="cuda", dtype=torch.float16):
    torch.manual_seed(0)
    gate = torch.randn(rows, intermediate_size, device=device, dtype=dtype)
    up = torch.randn(rows, intermediate_size, device=device, dtype=dtype)

    naive_mean, naive_std = _time_cuda(lambda: naive_silu_mul_reference(gate, up), n_trials)
    triton_mean, triton_std = _time_cuda(lambda: fused_silu_mul(gate, up), n_trials)
    return naive_mean, naive_std, triton_mean, triton_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256, 1024, 8192, 32768],
                        help="batch*seq_len flattened row count")
    parser.add_argument("--hidden-size", type=int, default=1536, help="Qwen2.5-1.5B-Instruct default")
    parser.add_argument("--intermediate-size", type=int, default=8960, help="Qwen2.5-1.5B-Instruct default")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/stage3_kernels.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    print(f"hidden_size={args.hidden_size} intermediate_size={args.intermediate_size} trials={args.trials}\n")

    print("=== RMSNorm ===")
    header = f"{'rows':>8}{'naive(us)':>14}{'triton(us)':>14}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    rmsnorm_results = []
    for rows in args.rows:
        naive_mean, naive_std, triton_mean, triton_std = benchmark_rmsnorm(rows, args.hidden_size, args.trials)
        speedup = naive_mean / triton_mean
        print(f"{rows:>8}{naive_mean*1000:>10.1f}±{naive_std*1000:<3.0f}{triton_mean*1000:>10.1f}±{triton_std*1000:<3.0f}{speedup:>9.2f}x")
        rmsnorm_results.append({
            "rows": rows, "hidden_size": args.hidden_size,
            "naive_ms_mean": naive_mean, "naive_ms_std": naive_std,
            "triton_ms_mean": triton_mean, "triton_ms_std": triton_std,
            "speedup": speedup,
        })

    print("\n=== SiLU-and-mul (SwiGLU activation) ===")
    print(header)
    print("-" * len(header))
    swiglu_results = []
    for rows in args.rows:
        naive_mean, naive_std, triton_mean, triton_std = benchmark_swiglu(rows, args.intermediate_size, args.trials)
        speedup = naive_mean / triton_mean
        print(f"{rows:>8}{naive_mean*1000:>10.1f}±{naive_std*1000:<3.0f}{triton_mean*1000:>10.1f}±{triton_std*1000:<3.0f}{speedup:>9.2f}x")
        swiglu_results.append({
            "rows": rows, "intermediate_size": args.intermediate_size,
            "naive_ms_mean": naive_mean, "naive_ms_std": naive_std,
            "triton_ms_mean": triton_mean, "triton_ms_std": triton_std,
            "speedup": speedup,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rmsnorm": rmsnorm_results, "swiglu": swiglu_results}, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote: these are small, memory-bound ops -- don't expect "
        "attention-sized speedups. The win here is fewer kernel launches "
        "and less memory traffic, which matters more in the full model "
        "(dozens of these calls per forward pass) than in any single "
        "isolated call."
    )


if __name__ == "__main__":
    main()
