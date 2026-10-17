"""
Stage 7 end-to-end benchmark: fp16 vs. INT8 weights.

One model instance: benchmarked as fp16, then quantized IN PLACE and
benchmarked again (same approach as Stage 3, and it guarantees both runs
use literally the same weights, just stored differently).

Two decode paths are measured, because they answer different questions:

1. HuggingFace path (BaselineEngine): the ordinary decode loop. At
   batch=1 this is dominated by Python/launch overhead (Stage 6's whole
   point), so halving weight bytes can only help as much as the GPU
   side matters -- expect a muted gain.
2. Graphed path (GraphedDecodeEngine, Stage 6): CUDA-graph replay strips
   out launch overhead, leaving the GPU memory traffic exposed. This is
   the cleanest test of "does halving weight bytes speed up a
   memory-bound decode step?" -- the stages compose here, which is the
   most informative number in this benchmark.

Two precautions worth recording:
- Triton JIT compilation of every kernel variant happens BEFORE graph
  capture (via a throwaway eager decode step). Compiling for the first
  time inside a capture region was never verified to be safe in Stage 6
  (its tests happened to run ungraphed code first); avoiding it entirely
  is cheaper than finding out.
- max_context_len is 512, not the engine's 4096 default. The paged
  attention kernel loops MAX_NUM_BLOCKS times with masking (a documented
  Stage 4 limitation), so a 4096 ceiling would burn wasted iterations on
  every step and dilute the very effect this benchmark is trying to
  measure. It affects fp16 and int8 equally, but dilution still hides
  the signal.

Usage:
    python benchmarks/run_stage7_e2e.py
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.baseline_engine import BaselineEngine
from src.engine.paged_engine import PagedEngine
from src.engine.graphed_decode_engine import GraphedDecodeEngine
from src.engine.profiler import InferenceProfiler
from src.kv_cache.block_manager import PagedKVCacheManager
from src.quantization.int8 import quantize_model_int8

BLOCK_SIZE = 16


def build_manager(config, max_pages):
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        block_size=BLOCK_SIZE, max_pages=max_pages, device="cuda", dtype=torch.float16,
    )


def layer_weight_bytes(model):
    """Bytes held by the transformer layers' parameters + buffers (the
    part this stage changes). Excludes embeddings/lm_head, which stay fp16."""
    total = 0
    for module in model.model.layers.modules():
        for p in module.parameters(recurse=False):
            total += p.numel() * p.element_size()
        for b in module.buffers(recurse=False):
            total += b.numel() * b.element_size()
    return total


def hf_decode_stats(loaded, seq_len, gen_len, batch, trials, warmup):
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device="cuda")
    profiler = InferenceProfiler(device="cuda")
    input_ids = torch.randint(0, loaded.tokenizer.vocab_size, (batch, seq_len), device="cuda")
    for _ in range(warmup):
        profiler.profile_generation(engine, input_ids, gen_len, "warmup")
    return profiler.profile_generation_repeated(engine, input_ids, gen_len, "measure", n_trials=trials)


def graphed_decode_tok_s(loaded, seq_len, gen_len, trials):
    config = loaded.model.config
    vocab = loaded.tokenizer.vocab_size

    # Precaution: compile every kernel variant eagerly before any capture.
    warm_manager = build_manager(config, max_pages=64)
    warm_engine = PagedEngine(loaded.model, loaded.tokenizer, warm_manager, device="cuda")
    warm_ids = torch.randint(0, vocab, (1, seq_len), device="cuda")
    warm_tok = warm_engine.prefill(warm_ids, seq_id=0)
    warm_engine.decode_step(warm_tok, seq_id=0)
    del warm_engine, warm_manager
    torch.cuda.synchronize()

    manager = build_manager(config, max_pages=512)
    engine = GraphedDecodeEngine(
        loaded.model, loaded.tokenizer, manager, max_context_len=512, device="cuda"
    )
    input_ids = torch.randint(0, vocab, (1, seq_len), device="cuda")

    tok = engine.prefill(input_ids, seq_id=0)
    engine.decode_step_graphed(tok, seq_id=0)  # triggers capture

    rates = []
    for t in range(trials):
        seq_id = t + 1
        cur = engine.prefill(input_ids, seq_id=seq_id)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(gen_len - 1):
            cur = engine.decode_step_graphed(cur, seq_id=seq_id)
        torch.cuda.synchronize()
        rates.append((gen_len - 1) / (time.perf_counter() - start))

    del engine, manager
    gc.collect()
    torch.cuda.empty_cache()
    return statistics.mean(rates), (statistics.stdev(rates) if len(rates) > 1 else 0.0)


def run_phase(label, loaded, args):
    print(f"\n=== {label} ===")
    weight_mb = layer_weight_bytes(loaded.model) / 1024**2
    print(f"  transformer-layer weights: {weight_mb:.0f} MB")

    phase = {"layer_weight_mb": weight_mb, "hf": {}}

    for batch in args.batch_sizes:
        r = hf_decode_stats(loaded, args.seq_len, args.gen_len, batch, args.trials, args.warmup)
        print(
            f"  HF path, batch={batch}: TTFT={r.ttft_ms_mean:.1f}ms  "
            f"decode={r.decode_tokens_per_sec_mean:.1f}±{r.decode_tokens_per_sec_std:.1f} tok/s  "
            f"peak={r.peak_memory_mb_mean:.0f}MB"
        )
        phase["hf"][batch] = {
            "ttft_ms": r.ttft_ms_mean,
            "decode_tok_s": r.decode_tokens_per_sec_mean,
            "peak_mb": r.peak_memory_mb_mean,
        }

    g_mean, g_std = graphed_decode_tok_s(loaded, args.seq_len, args.gen_len, args.trials)
    print(f"  Graphed path, batch=1: decode={g_mean:.1f}±{g_std:.1f} tok/s")
    phase["graphed_decode_tok_s"] = g_mean
    return phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/stage7_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: requires a CUDA GPU.")
        sys.exit(1)

    print(f"Loading {args.model} [sdpa, fp16]...")
    loaded = load_model(args.model, device="cuda", attn_implementation="sdpa")

    fp16 = run_phase("fp16 weights", loaded, args)

    stats = quantize_model_int8(loaded.model)
    print(
        f"\nQuantized {stats['layers_quantized']} linear layers "
        f"({stats['bytes_before']/1024**2:.0f} MB -> {stats['bytes_after']/1024**2:.0f} MB)"
    )
    if stats["layers_quantized"] == 0:
        print("WARNING: nothing was quantized -- the 'int8' numbers below are really fp16.")

    int8 = run_phase("int8 weights", loaded, args)

    print("\n=== Comparison ===")
    print(f"  weights: {fp16['layer_weight_mb']:.0f} MB -> {int8['layer_weight_mb']:.0f} MB "
          f"({fp16['layer_weight_mb']/int8['layer_weight_mb']:.2f}x smaller)")
    for batch in args.batch_sizes:
        f, i = fp16["hf"][batch], int8["hf"][batch]
        print(
            f"  HF path batch={batch}: decode {f['decode_tok_s']:.1f} -> {i['decode_tok_s']:.1f} tok/s "
            f"({i['decode_tok_s']/f['decode_tok_s']:.2f}x), peak mem {f['peak_mb']:.0f} -> {i['peak_mb']:.0f} MB"
        )
    print(
        f"  Graphed path: decode {fp16['graphed_decode_tok_s']:.1f} -> {int8['graphed_decode_tok_s']:.1f} tok/s "
        f"({int8['graphed_decode_tok_s']/fp16['graphed_decode_tok_s']:.2f}x)"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fp16": fp16, "int8": int8, "quantize_stats": stats}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
