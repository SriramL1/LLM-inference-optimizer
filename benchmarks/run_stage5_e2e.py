"""
Stage 5 throughput benchmark.

Compares total wall-clock time to process a fixed set of requests two
ways: sequentially (one full request finishes before the next starts --
what you'd get without continuous batching, even if using Stage 4's
paged cache) versus via the ContinuousBatchingScheduler (multiple
sequences' decode steps batched together, new requests admitted as soon
as a slot frees up).

This is the number that actually demonstrates Stage 5's value -- Stage
1 already showed decode throughput scales near-linearly with batch size
in isolation; this shows that scaling holds up in a scheduler that
handles requests arriving with different lengths, not just a
pre-formed, uniform-length batch.

Usage:
    python benchmarks/run_stage5_e2e.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.paged_engine import PagedEngine
from src.engine.continuous_batching_engine import ContinuousBatchingEngine
from src.engine.scheduler import ContinuousBatchingScheduler
from src.kv_cache.block_manager import PagedKVCacheManager

BLOCK_SIZE = 16


def build_manager(config, max_pages=4096):
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=BLOCK_SIZE, max_pages=max_pages, device="cuda", dtype=torch.float16,
    )


def build_requests(tokenizer, num_requests, min_len, max_len, gen_len, device="cuda"):
    """Synthetic requests with varying prompt lengths -- uniform-length
    requests would understate continuous batching's real advantage,
    since real traffic never arrives at uniform length."""
    torch.manual_seed(0)
    requests = []
    for i in range(num_requests):
        seq_len = min_len + (i * (max_len - min_len) // max(1, num_requests - 1))
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
        requests.append((input_ids, gen_len))
    return requests


def run_sequential(loaded, requests):
    total_tokens = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for input_ids, gen_len in requests:
        manager = build_manager(loaded.model.config)
        engine = PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
        engine.generate(input_ids, max_new_tokens=gen_len, seq_id=0)
        total_tokens += gen_len
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, total_tokens


def run_continuous_batching(loaded, requests, max_batch_size):
    manager = build_manager(loaded.model.config)
    engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    scheduler = ContinuousBatchingScheduler(engine, max_batch_size=max_batch_size)

    total_tokens = sum(gen_len for _, gen_len in requests)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for input_ids, gen_len in requests:
        scheduler.add_request(input_ids, gen_len)
    scheduler.run_to_completion()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--min-len", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--out", default="benchmarks/results/stage5_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    print(f"Loading {args.model} [sdpa]...")
    loaded = load_model(args.model, device="cuda", attn_implementation="sdpa")

    requests = build_requests(
        loaded.tokenizer, args.num_requests, args.min_len, args.max_len, args.gen_len
    )
    print(
        f"{args.num_requests} requests, prompt lengths {args.min_len}-{args.max_len}, "
        f"{args.gen_len} tokens generated each\n"
    )

    print("Running sequentially (no continuous batching)...")
    seq_elapsed, seq_tokens = run_sequential(loaded, requests)
    seq_throughput = seq_tokens / seq_elapsed
    print(f"  {seq_elapsed:.2f}s total, {seq_throughput:.1f} tok/s aggregate throughput")

    print(f"\nRunning via ContinuousBatchingScheduler (max_batch_size={args.max_batch_size})...")
    cb_elapsed, cb_tokens = run_continuous_batching(loaded, requests, args.max_batch_size)
    cb_throughput = cb_tokens / cb_elapsed
    print(f"  {cb_elapsed:.2f}s total, {cb_throughput:.1f} tok/s aggregate throughput")

    speedup = seq_elapsed / cb_elapsed
    print(f"\nSpeedup: {speedup:.2f}x")

    result = {
        "num_requests": args.num_requests,
        "min_len": args.min_len, "max_len": args.max_len, "gen_len": args.gen_len,
        "max_batch_size": args.max_batch_size,
        "sequential_elapsed_s": seq_elapsed, "sequential_throughput_tok_s": seq_throughput,
        "continuous_batching_elapsed_s": cb_elapsed, "continuous_batching_throughput_tok_s": cb_throughput,
        "speedup": speedup,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
