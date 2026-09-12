"""
Stage 3 end-to-end benchmark.

Unlike Stage 2's benchmark (which had to load the model twice, since
attn_implementation is fixed at load time), this one loads the model
once and benchmarks it before and after patching in-place -- simpler,
and also guarantees both runs use the literal same weights, not just the
same weights loaded twice.

Usage:
    python benchmarks/run_stage3_e2e.py
    python benchmarks/run_stage3_e2e.py --seq-lens 512 2048 --trials 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.engine.model_loader import load_model
from src.engine.baseline_engine import BaselineEngine
from src.engine.profiler import InferenceProfiler
from src.engine.fused_norm_activation_patch import patch_model_with_fused_kernels


def build_batch(vocab_size, seq_len, batch_size, device):
    return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--attn-implementation", default="sdpa",
                        help="'eager' is known to produce NaN in some torch/transformers/GPU combinations "
                             "(see docs/stage3-fused-norm-activation.md) -- default to 'sdpa'. Use "
                             "'triton_attn' to measure Stage 2+3 stacked together.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/stage3_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    device = "cuda"
    if args.attn_implementation == "triton_attn":
        from src.engine.flash_attention_patch import register_triton_flash_attention
        register_triton_flash_attention()

    print(f"Loading {args.model} [attn_implementation={args.attn_implementation}]...")
    loaded = load_model(args.model, device=device, attn_implementation=args.attn_implementation)
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device=device)
    profiler = InferenceProfiler(device=device)

    print("\n=== Before Stage 3 patch ===")
    before_results = []
    for seq_len in args.seq_lens:
        input_ids = build_batch(loaded.tokenizer.vocab_size, seq_len, args.batch_size, device)
        for _ in range(args.warmup):
            profiler.profile_generation(engine, input_ids, args.gen_len, args.model)
        r = profiler.profile_generation_repeated(engine, input_ids, args.gen_len, "before", n_trials=args.trials)
        before_results.append(r)
        print(f"  seq_len={seq_len:<6} TTFT={r.ttft_ms_mean:.1f}±{r.ttft_ms_std:.1f}ms  decode={r.decode_tokens_per_sec_mean:.1f}±{r.decode_tokens_per_sec_std:.1f} tok/s")

    counts = patch_model_with_fused_kernels(loaded.model)
    print(f"\nPatched {counts['rmsnorm']} RMSNorm modules, {counts['mlp']} MLP modules.")
    if counts["rmsnorm"] == 0 or counts["mlp"] == 0:
        print("WARNING: one or both patch counts are zero -- the 'after' numbers below "
              "won't reflect any real change. Check class name matching in "
              "fused_norm_activation_patch.py against this model's actual module names.")

    print("\n=== After Stage 3 patch ===")
    after_results = []
    for seq_len in args.seq_lens:
        input_ids = build_batch(loaded.tokenizer.vocab_size, seq_len, args.batch_size, device)
        for _ in range(args.warmup):
            profiler.profile_generation(engine, input_ids, args.gen_len, args.model)
        r = profiler.profile_generation_repeated(engine, input_ids, args.gen_len, "after", n_trials=args.trials)
        after_results.append(r)
        print(f"  seq_len={seq_len:<6} TTFT={r.ttft_ms_mean:.1f}±{r.ttft_ms_std:.1f}ms  decode={r.decode_tokens_per_sec_mean:.1f}±{r.decode_tokens_per_sec_std:.1f} tok/s")

    print("\n=== Comparison ===")
    header = f"{'seq_len':>8}{'before TTFT':>16}{'after TTFT':>16}{'TTFT speedup':>14}{'decode speedup':>16}"
    print(header)
    print("-" * len(header))
    all_out = []
    for b, a in zip(before_results, after_results):
        ttft_speedup = b.ttft_ms_mean / a.ttft_ms_mean
        decode_speedup = a.decode_tokens_per_sec_mean / b.decode_tokens_per_sec_mean if b.decode_tokens_per_sec_mean > 0 else float("nan")
        print(f"{b.prompt_tokens:>8}{b.ttft_ms_mean:>12.1f}ms{a.ttft_ms_mean:>12.1f}ms{ttft_speedup:>13.2f}x{decode_speedup:>15.2f}x")
        all_out.append({
            "seq_len": b.prompt_tokens,
            "before_ttft_ms_mean": b.ttft_ms_mean, "after_ttft_ms_mean": a.ttft_ms_mean,
            "ttft_speedup": ttft_speedup,
            "before_decode_tok_s": b.decode_tokens_per_sec_mean, "after_decode_tok_s": a.decode_tokens_per_sec_mean,
            "decode_speedup": decode_speedup,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rmsnorm_patched": counts["rmsnorm"], "mlp_patched": counts["mlp"], "results": all_out}, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote: unlike Stage 2, decode SHOULD improve here too (not just "
        "TTFT) -- RMSNorm and the MLP activation run during every decode "
        "step as well as prefill, so this stage's kernels apply to both "
        "phases. Don't expect a large speedup either way, though: these "
        "are a small fraction of total per-layer compute compared to the "
        "matmuls in attention and the MLP projections."
    )


if __name__ == "__main__":
    main()
