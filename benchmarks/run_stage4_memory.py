"""
Stage 4 memory-efficiency comparison.

The speed benchmark (run_stage4_kernel.py) deliberately doesn't show
paging's actual point -- it compares uniform-length batches, where a
plain contiguous cache is perfectly fine. This script shows the case
paging actually exists for: a batch of sequences with wildly different
lengths, where a naive system has to pre-allocate every sequence's cache
for some worst-case max length, wasting memory on every shorter
sequence -- versus paging, which only ever allocates what's used,
rounded up to the nearest page.

This is pure allocation-policy arithmetic, not a kernel benchmark, so it
uses the lightweight BlockAllocator/BlockTable directly rather than
PagedKVCacheManager (which eagerly allocates real backing tensors sized
for the whole page pool -- fine for actual inference, wildly wasteful
just to count bytes for a demo).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kv_cache.block_manager import BlockAllocator, BlockTable

NUM_LAYERS = 28       # Qwen2.5-1.5B-Instruct
NUM_KV_HEADS = 2
HEAD_DIM = 128
DTYPE_BYTES = 2       # fp16


def bytes_per_token():
    return NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * 2 * DTYPE_BYTES  # K and V


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[50, 200, 800, 30, 1500, 60],
        help="a realistic mixed-length batch, e.g. a real serving workload",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096,
        help="worst-case length a naive contiguous allocator must reserve per sequence",
    )
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    bpt = bytes_per_token()

    print(f"Batch: {len(args.seq_lens)} sequences, lengths={args.seq_lens}")
    print(f"Bytes per token (all layers, K+V): {bpt}")
    print()

    # Naive: every sequence reserves max_seq_len worth of memory,
    # regardless of its actual length.
    naive_bytes = len(args.seq_lens) * args.max_seq_len * bpt

    # Paged: each sequence uses only ceil(len / block_size) pages,
    # simulated with the lightweight allocator (no real tensors).
    total_pages_needed = sum(
        (seq_len + args.block_size - 1) // args.block_size for seq_len in args.seq_lens
    )
    allocator = BlockAllocator(total_pages_needed)
    tables = []
    for seq_len in args.seq_lens:
        table = BlockTable()
        num_blocks = (seq_len + args.block_size - 1) // args.block_size
        for _ in range(num_blocks):
            table.page_ids.append(allocator.allocate())
        table.context_len = seq_len
        tables.append(table)

    used_pages = total_pages_needed  # every allocated page is "used" in this static snapshot
    paged_bytes = used_pages * args.block_size * bpt

    def fmt_mb(b):
        return f"{b / (1024 ** 2):.1f} MB"

    print(f"{'Naive (pre-allocate max_seq_len per sequence)':<50}{fmt_mb(naive_bytes):>12}")
    print(f"{'Paged (allocate only what is used, per block)':<50}{fmt_mb(paged_bytes):>12}")
    print(f"{'Memory saved':<50}{fmt_mb(naive_bytes - paged_bytes):>12}")
    print(f"{'Reduction':<50}{(1 - paged_bytes / naive_bytes) * 100:>11.1f}%")
    print()
    print(
        "This is paging's actual payoff: the naive approach wastes memory "
        "proportional to (max_seq_len - actual_len) for every sequence "
        "shorter than the worst case. Paging only wastes, at most, "
        "(block_size - 1) tokens per sequence (partial last page) -- "
        "orders of magnitude less for realistic length distributions."
    )


if __name__ == "__main__":
    main()
