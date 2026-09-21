"""
Stage 6 benchmark: decode throughput, graphed vs. ungraphed.

Times decode steps only (not the one-time warmup+capture cost, reported
separately for honesty -- it's real setup latency, paid once per engine
lifetime, not per generated token). This isolates exactly the thing
Stage 6 is meant to improve: per-step Python/launch overhead, which
should shrink to near-zero once a graph is captured and simply replayed.

Usage:
    python benchmarks/run_stage6_e2e.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.paged_engine import PagedEngine
from src.engine.graphed_decode_engine import GraphedDecodeEngine
from src.kv_cache.block_manager import PagedKVCacheManager

BLOCK_SIZE = 16


def build_manager(config, max_pages=4096):
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=BLOCK_SIZE, max_pages=max_pages, device="cuda", dtype=torch.float16,
    )


def time_ungraphed(loaded, seq_len, gen_len, trials):
    input_ids = torch.randint(0, loaded.tokenizer.vocab_size, (1, seq_len), device="cuda")
    times = []
    for trial in range(trials):
        manager = build_manager(loaded.model.config)
        engine = PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
        next_token = engine.prefill(input_ids, seq_id=0)

        torch.cuda.synchronize()
        start = time.perf_counter()
        cur = next_token
        for _ in range(gen_len - 1):
            cur = engine.decode_step(cur, seq_id=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    mean = statistics.mean(times)
    tok_s = (gen_len - 1) / mean
    return mean, tok_s


def time_graphed(loaded, seq_len, gen_len, trials):
    input_ids = torch.randint(0, loaded.tokenizer.vocab_size, (1, seq_len), device="cuda")

    manager = build_manager(loaded.model.config)
    engine = GraphedDecodeEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    next_token = engine.prefill(input_ids, seq_id=0)

    # First decode call triggers warmup + capture -- time it separately.
    torch.cuda.synchronize()
    capture_start = time.perf_counter()
    cur = engine.decode_step_graphed(next_token, seq_id=0)
    torch.cuda.synchronize()
    capture_time_s = time.perf_counter() - capture_start

    # Remaining decode steps of this first generation are pure replays.
    times = []
    for _ in range(gen_len - 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        cur = engine.decode_step_graphed(cur, seq_id=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    # Additional trials: fresh SEQUENCES (new seq_id) within the SAME
    # manager, reusing the SAME captured graph -- pure replay, no
    # capture cost. Deliberately NOT a fresh PagedKVCacheManager per
    # trial: a captured graph is bound to the specific K/V cache tensor
    # memory addresses that existed at capture time, so swapping the
    # manager afterward would silently still write into the ORIGINAL
    # manager's tensors, not a new one's -- a real bug caught while
    # writing this benchmark, not a hypothetical one.
    for trial in range(trials - 1):
        seq_id = trial + 100
        next_token2 = engine.prefill(input_ids, seq_id=seq_id)
        cur2 = next_token2
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(gen_len - 1):
            cur2 = engine.decode_step_graphed(cur2, seq_id=seq_id)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / (gen_len - 1))

    per_step_mean = statistics.mean(times)
    tok_s = 1.0 / per_step_mean
    return capture_time_s, per_step_mean * (gen_len - 1), tok_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="benchmarks/results/stage6_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    print(f"Loading {args.model} [sdpa]...")
    loaded = load_model(args.model, device="cuda", attn_implementation="sdpa")

    print(f"\nseq_len={args.seq_len}, gen_len={args.gen_len}, trials={args.trials}\n")

    print("Ungraphed (PagedEngine.decode_step)...")
    ungraphed_mean_s, ungraphed_tok_s = time_ungraphed(loaded, args.seq_len, args.gen_len, args.trials)
    print(f"  {ungraphed_mean_s*1000:.1f}ms per generation, {ungraphed_tok_s:.1f} tok/s")

    print("\nGraphed (GraphedDecodeEngine.decode_step_graphed)...")
    capture_time_s, replay_total_s, graphed_tok_s = time_graphed(loaded, args.seq_len, args.gen_len, args.trials)
    print(f"  one-time warmup+capture: {capture_time_s*1000:.1f}ms")
    print(f"  steady-state replay: {graphed_tok_s:.1f} tok/s")

    speedup = graphed_tok_s / ungraphed_tok_s
    print(f"\nSteady-state decode speedup: {speedup:.2f}x")

    result = {
        "seq_len": args.seq_len, "gen_len": args.gen_len,
        "ungraphed_tok_s": ungraphed_tok_s,
        "graphed_capture_time_ms": capture_time_s * 1000,
        "graphed_steady_state_tok_s": graphed_tok_s,
        "speedup": speedup,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
