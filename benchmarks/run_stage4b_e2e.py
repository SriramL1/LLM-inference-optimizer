"""
Stage 4b end-to-end benchmark.

Honest framing: PagedEngine's prefill reuses Stage 2's flash_attention
kernel (not SDPA/eager), so this benchmark measures Stages 2 and 4
working together against the Stage 1 baseline -- not Stage 4 in
isolation. That's an accurate reflection of what PagedEngine actually
is (a from-scratch engine assembled from this project's own kernels),
just worth being explicit about when reading the numbers: don't
attribute the whole delta to paging alone.

Also reports real memory usage via the manager's own accounting, since
that -- not raw decode speed -- is Stage 4's actual point.

Usage:
    python benchmarks/run_stage4b_e2e.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.baseline_engine import BaselineEngine
from src.engine.paged_engine import PagedEngine
from src.engine.profiler import InferenceProfiler
from src.kv_cache.block_manager import PagedKVCacheManager

BLOCK_SIZE = 16


def build_manager(config, max_pages=4096):
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        max_pages=max_pages,
        device="cuda",
        dtype=torch.float16,
    )


def time_baseline(loaded, seq_len, gen_len, trials, warmup):
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device="cuda")
    profiler = InferenceProfiler(device="cuda")
    input_ids = torch.randint(0, loaded.tokenizer.vocab_size, (1, seq_len), device="cuda")
    for _ in range(warmup):
        profiler.profile_generation(engine, input_ids, gen_len, "baseline")
    return profiler.profile_generation_repeated(engine, input_ids, gen_len, "baseline", n_trials=trials)


def time_paged(loaded, seq_len, gen_len, trials, warmup):
    import statistics
    input_ids = torch.randint(0, loaded.tokenizer.vocab_size, (1, seq_len), device="cuda")

    def run_once(seq_id):
        manager = build_manager(loaded.model.config)
        engine = PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        next_token = engine.prefill(input_ids, seq_id)
        end.record()
        torch.cuda.synchronize()
        ttft_ms = start.elapsed_time(end)

        decode_start, decode_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        decode_start.record()
        cur = next_token
        for _ in range(gen_len - 1):
            cur = engine.decode_step(cur, seq_id)
        decode_end.record()
        torch.cuda.synchronize()
        decode_ms = decode_start.elapsed_time(decode_end)
        decode_tok_s = (gen_len - 1) / (decode_ms / 1000) if decode_ms > 0 else 0.0
        return ttft_ms, decode_tok_s, manager.memory_usage()

    for i in range(warmup):
        run_once(seq_id=1000 + i)

    ttfts, decode_tps = [], []
    mem_snapshot = None
    for i in range(trials):
        ttft, dtps, mem = run_once(seq_id=2000 + i)
        ttfts.append(ttft)
        decode_tps.append(dtps)
        mem_snapshot = mem

    return {
        "ttft_ms_mean": statistics.mean(ttfts),
        "ttft_ms_std": statistics.stdev(ttfts) if trials > 1 else 0.0,
        "decode_tok_s_mean": statistics.mean(decode_tps),
        "decode_tok_s_std": statistics.stdev(decode_tps) if trials > 1 else 0.0,
        "memory": mem_snapshot,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/stage4b_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    print(f"Loading {args.model} [sdpa baseline]...")
    loaded = load_model(args.model, device="cuda", attn_implementation="sdpa")

    results = []
    for seq_len in args.seq_lens:
        print(f"\n--- seq_len={seq_len} ---")
        baseline = time_baseline(loaded, seq_len, args.gen_len, args.trials, args.warmup)
        print(
            f"baseline (sdpa):  TTFT={baseline.ttft_ms_mean:.1f}±{baseline.ttft_ms_std:.1f}ms  "
            f"decode={baseline.decode_tokens_per_sec_mean:.1f}±{baseline.decode_tokens_per_sec_std:.1f} tok/s"
        )

        paged = time_paged(loaded, seq_len, args.gen_len, args.trials, args.warmup)
        print(
            f"PagedEngine:      TTFT={paged['ttft_ms_mean']:.1f}±{paged['ttft_ms_std']:.1f}ms  "
            f"decode={paged['decode_tok_s_mean']:.1f}±{paged['decode_tok_s_std']:.1f} tok/s"
        )
        mem = paged["memory"]
        print(
            f"PagedEngine cache memory: {mem['used_bytes'] / 1024**2:.1f}MB used "
            f"/ {mem['total_bytes'] / 1024**2:.1f}MB pool ({mem['used_pages']}/{mem['total_pages']} pages)"
        )

        results.append({
            "seq_len": seq_len,
            "baseline_ttft_ms": baseline.ttft_ms_mean,
            "baseline_decode_tok_s": baseline.decode_tokens_per_sec_mean,
            "paged_ttft_ms": paged["ttft_ms_mean"],
            "paged_decode_tok_s": paged["decode_tok_s_mean"],
            "paged_memory": mem,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote: PagedEngine's prefill reuses Stage 2's flash attention "
        "kernel, not SDPA/eager -- this compares Stages 2+4 combined "
        "against the Stage 1 baseline, not Stage 4 in isolation. Memory "
        "usage (not decode speed) is this stage's actual point."
    )


if __name__ == "__main__":
    main()
