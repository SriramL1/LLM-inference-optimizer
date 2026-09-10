"""
Stage 2 attention microbenchmark.

Compares three implementations across a sweep of sequence lengths:
  1. naive       -- unfused PyTorch (materializes the full attention matrix)
  2. sdpa         -- torch.nn.functional.scaled_dot_product_attention
                     (PyTorch's own fused/flash backend -- the real bar to
                     clear, not just "faster than naive")
  3. triton       -- our Stage 2 kernel

This benchmarks the attention op in isolation (not the full model), on
shapes matching Qwen2.5-1.5B-Instruct (12 Q heads, 2 KV heads, head_dim
128) by default. The point of Stage 2 is prefill (compute-bound, long
sequences) -- so unlike Stage 1's benchmark, the interesting axis here is
sequence length, not batch size.

Usage:
    python benchmarks/run_stage2_attention.py
    python benchmarks/run_stage2_attention.py --seq-lens 128 512 2048 --trials 10
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.kernels.flash_attention_triton import flash_attention, naive_attention_reference


def _time_cuda(fn, n_trials: int, warmup: int = 3):
    """Times fn() with CUDA events, returns (mean_ms, std_ms)."""
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

    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if n_trials > 1 else 0.0
    return mean, std


def benchmark_config(batch, num_q_heads, num_kv_heads, seq_len, head_dim, n_trials, device="cuda", dtype=torch.float16):
    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1

    if num_kv_heads != num_q_heads:
        rep = num_q_heads // num_kv_heads
        k_expanded = k.repeat_interleave(rep, dim=1)
        v_expanded = v.repeat_interleave(rep, dim=1)
    else:
        k_expanded, v_expanded = k, v

    results = {}

    naive_mean, naive_std = _time_cuda(
        lambda: naive_attention_reference(q, k, v, causal=True), n_trials
    )
    results["naive"] = (naive_mean, naive_std)

    sdpa_mean, sdpa_std = _time_cuda(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k_expanded, v_expanded, is_causal=True
        ),
        n_trials,
    )
    results["sdpa"] = (sdpa_mean, sdpa_std)

    triton_mean, triton_std = _time_cuda(
        lambda: flash_attention(q, k, v, causal=True), n_trials
    )
    results["triton"] = (triton_mean, triton_std)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-q-heads", type=int, default=12, help="Qwen2.5-1.5B-Instruct default")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="Qwen2.5-1.5B-Instruct default (GQA)")
    parser.add_argument("--head-dim", type=int, default=128, help="Qwen2.5-1.5B-Instruct default")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", default="benchmarks/results/stage2_attention.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    print(
        f"batch={args.batch} q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} trials={args.trials}\n"
    )
    header = f"{'seq_len':>8}{'naive(ms)':>14}{'sdpa(ms)':>14}{'triton(ms)':>14}{'vs sdpa':>10}{'vs naive':>10}"
    print(header)
    print("-" * len(header))

    all_results = []
    for seq_len in args.seq_lens:
        r = benchmark_config(
            args.batch, args.num_q_heads, args.num_kv_heads, seq_len, args.head_dim, args.trials
        )
        naive_mean, naive_std = r["naive"]
        sdpa_mean, sdpa_std = r["sdpa"]
        triton_mean, triton_std = r["triton"]

        speedup_vs_sdpa = sdpa_mean / triton_mean
        speedup_vs_naive = naive_mean / triton_mean

        print(
            f"{seq_len:>8}"
            f"{naive_mean:>9.2f}±{naive_std:<4.2f}"
            f"{sdpa_mean:>9.2f}±{sdpa_std:<4.2f}"
            f"{triton_mean:>9.2f}±{triton_std:<4.2f}"
            f"{speedup_vs_sdpa:>9.2f}x"
            f"{speedup_vs_naive:>9.2f}x"
        )

        all_results.append({
            "seq_len": seq_len,
            "batch": args.batch,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "n_trials": args.trials,
            "naive_ms_mean": naive_mean, "naive_ms_std": naive_std,
            "sdpa_ms_mean": sdpa_mean, "sdpa_ms_std": sdpa_std,
            "triton_ms_mean": triton_mean, "triton_ms_std": triton_std,
            "speedup_vs_sdpa": speedup_vs_sdpa,
            "speedup_vs_naive": speedup_vs_naive,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote: beating 'naive' is the easy bar (it materializes the full "
        "attention matrix). Beating or matching 'sdpa' is the real one -- "
        "that's PyTorch's own production flash-attention backend."
    )


if __name__ == "__main__":
    main()
