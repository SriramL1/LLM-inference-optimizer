"""
Stage 4 isolated kernel microbenchmark.

Compares the paged-attention decode kernel against a plain SDPA decode
step over an equivalent, ordinary contiguous KV-cache -- same context
length, same data, only the memory layout differs (paged/gathered vs.
one flat tensor).

Honest framing: this benchmark is expected to show the paged kernel is
roughly on par with or somewhat SLOWER than the contiguous-cache
baseline at a given context length, since gathering scattered pages via
a block table is inherently more work than reading one flat tensor.
That's not a failure -- it's the same tradeoff vLLM's own paper reports.
Paging's payoff is memory efficiency and the ability to batch
variable-length sequences without padding waste, which this isolated,
uniform-length benchmark doesn't even exercise. See
docs/stage4-paged-kv-cache.md for the fuller picture, including a
separate memory-usage comparison.

Usage:
    python benchmarks/run_stage4_kernel.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.kv_cache.block_manager import PagedKVCacheManager
from src.kernels.paged_attention_triton import paged_attention_decode

BLOCK_SIZE = 16
HEAD_DIM = 128
NUM_KV_HEADS = 2
NUM_Q_HEADS = 12


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


def build_paged_cache(batch, context_len, block_size=BLOCK_SIZE):
    torch.manual_seed(0)
    manager = PagedKVCacheManager(
        num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=block_size, max_pages=4096, device="cuda", dtype=torch.float16,
    )
    seq_ids = list(range(batch))
    for seq_id in seq_ids:
        manager.allocate_sequence(seq_id)
        k = torch.randn(context_len, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
        v = torch.randn(context_len, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
        positions = manager.reserve(seq_id, context_len)
        manager.write(seq_id, layer_idx=0, k=k, v=v, positions=positions)

    max_blocks = (context_len + block_size - 1) // block_size
    block_tables = manager.batch_block_table_tensor(seq_ids, max_blocks)
    context_lens_tensor = manager.batch_context_lens_tensor(seq_ids)
    return manager, block_tables, context_lens_tensor


def benchmark_config(batch, context_len, n_trials):
    manager, block_tables, context_lens_tensor = build_paged_cache(batch, context_len)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1

    paged_mean, paged_std = _time_cuda(
        lambda: paged_attention_decode(
            q, manager.k_cache[0], manager.v_cache[0], block_tables, context_lens_tensor, BLOCK_SIZE
        ),
        n_trials,
    )

    # Contiguous-cache baseline: same K/V values, laid out as one flat
    # tensor per sequence (what a plain DynamicCache would look like),
    # then a standard SDPA decode step.
    k_contig = torch.randn(batch, NUM_KV_HEADS, context_len, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
    v_contig = torch.randn(batch, NUM_KV_HEADS, context_len, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
    q_sdpa = q.unsqueeze(2)  # (batch, num_q_heads, 1, head_dim)
    heads_per_group = NUM_Q_HEADS // NUM_KV_HEADS
    k_expanded = k_contig.repeat_interleave(heads_per_group, dim=1)
    v_expanded = v_contig.repeat_interleave(heads_per_group, dim=1)

    sdpa_mean, sdpa_std = _time_cuda(
        lambda: torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_expanded, v_expanded),
        n_trials,
    )

    return paged_mean, paged_std, sdpa_mean, sdpa_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/stage4_kernel.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    header = f"{'batch':>7}{'ctx_len':>9}{'paged(us)':>14}{'sdpa(us)':>14}{'paged/sdpa':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for batch in args.batch_sizes:
        for ctx_len in args.context_lens:
            paged_mean, paged_std, sdpa_mean, sdpa_std = benchmark_config(batch, ctx_len, args.trials)
            ratio = paged_mean / sdpa_mean
            print(
                f"{batch:>7}{ctx_len:>9}"
                f"{paged_mean*1000:>10.1f}±{paged_std*1000:<3.0f}"
                f"{sdpa_mean*1000:>10.1f}±{sdpa_std*1000:<3.0f}"
                f"{ratio:>11.2f}x"
            )
            results.append({
                "batch": batch, "context_len": ctx_len,
                "paged_ms_mean": paged_mean, "paged_ms_std": paged_std,
                "sdpa_ms_mean": sdpa_mean, "sdpa_ms_std": sdpa_std,
                "paged_over_sdpa_ratio": ratio,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote: paged/sdpa > 1.0x means the paged kernel is SLOWER per-call "
        "than a contiguous-cache baseline -- expected, due to gathered "
        "memory access. This benchmark deliberately does not show paging's "
        "actual payoff (memory efficiency, variable-length batching without "
        "padding waste) -- see docs/stage4-paged-kv-cache.md."
    )


if __name__ == "__main__":
    main()
