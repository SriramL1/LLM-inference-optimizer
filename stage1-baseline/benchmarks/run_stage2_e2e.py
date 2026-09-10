"""
Stage 2b end-to-end benchmark.

The isolated kernel benchmark (run_stage2_attention.py) showed the Triton
kernel matches SDPA within noise at long sequences -- but that's the
attention op alone. Amdahl's Law means that doesn't automatically
translate into a proportional end-to-end speedup: the rest of the model
(MLP layers, layernorm, embedding, LM head) doesn't get any faster, so
however big a fraction of prefill time attention was, that's roughly the
ceiling on the whole-model speedup. This script measures the number that
actually matters: real TTFT and decode throughput, full model, eager vs
Triton-fused-attention.

Usage:
    python benchmarks/run_stage2_e2e.py
    python benchmarks/run_stage2_e2e.py --seq-lens 512 2048 4096 --trials 8
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
from src.engine.flash_attention_patch import register_triton_flash_attention, ATTN_IMPL_NAME


def build_batch(vocab_size: int, seq_len: int, batch_size: int, device: str) -> torch.Tensor:
    # Random token ids -- content doesn't matter for a timing benchmark,
    # only shape does. Avoids needing a real long prompt.
    return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device)


def run_variant(model_name, attn_impl, seq_lens, batch_size, gen_len, trials, warmup, device):
    print(f"Loading {model_name} [attn_implementation={attn_impl}]...")
    loaded = load_model(model_name, device=device, attn_implementation=attn_impl)
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device=device)
    profiler = InferenceProfiler(device=device)

    results = []
    for seq_len in seq_lens:
        input_ids = build_batch(loaded.tokenizer.vocab_size, seq_len, batch_size, device)
        for _ in range(warmup):
            profiler.profile_generation(engine, input_ids, gen_len, model_name)

        result = profiler.profile_generation_repeated(
            engine, input_ids, gen_len, f"{model_name} [{attn_impl}]", n_trials=trials
        )
        results.append(result)
        print(
            f"  seq_len={seq_len:<6} TTFT={result.ttft_ms_mean:.1f}±{result.ttft_ms_std:.1f}ms  "
            f"decode={result.decode_tokens_per_sec_mean:.1f}±{result.decode_tokens_per_sec_std:.1f} tok/s"
        )

    # Free GPU memory before loading the next variant -- both models
    # loaded simultaneously would roughly double peak VRAM usage.
    del loaded, engine
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=16, help="kept short -- this benchmark is about TTFT/prefill, not decode")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/stage2_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark requires a CUDA GPU.")
        sys.exit(1)

    device = "cuda"
    register_triton_flash_attention()

    print("=== Eager attention (Stage 1 baseline) ===")
    eager_results = run_variant(
        args.model, "eager", args.seq_lens, args.batch_size, args.gen_len, args.trials, args.warmup, device
    )

    print("\n=== Triton fused attention (Stage 2) ===")
    triton_results = run_variant(
        args.model, ATTN_IMPL_NAME, args.seq_lens, args.batch_size, args.gen_len, args.trials, args.warmup, device
    )

    print("\n=== End-to-end comparison ===")
    header = f"{'seq_len':>8}{'eager TTFT':>16}{'triton TTFT':>16}{'TTFT speedup':>14}{'decode speedup':>16}"
    print(header)
    print("-" * len(header))

    all_out = []
    for eager_r, triton_r in zip(eager_results, triton_results):
        ttft_speedup = eager_r.ttft_ms_mean / triton_r.ttft_ms_mean
        decode_speedup = (
            triton_r.decode_tokens_per_sec_mean / eager_r.decode_tokens_per_sec_mean
            if eager_r.decode_tokens_per_sec_mean > 0 else float("nan")
        )
        print(
            f"{eager_r.prompt_tokens:>8}"
            f"{eager_r.ttft_ms_mean:>12.1f}ms"
            f"{triton_r.ttft_ms_mean:>12.1f}ms"
            f"{ttft_speedup:>13.2f}x"
            f"{decode_speedup:>15.2f}x"
        )
        all_out.append({
            "seq_len": eager_r.prompt_tokens,
            "eager_ttft_ms_mean": eager_r.ttft_ms_mean,
            "eager_ttft_ms_std": eager_r.ttft_ms_std,
            "triton_ttft_ms_mean": triton_r.ttft_ms_mean,
            "triton_ttft_ms_std": triton_r.ttft_ms_std,
            "ttft_speedup": ttft_speedup,
            "eager_decode_tok_s": eager_r.decode_tokens_per_sec_mean,
            "triton_decode_tok_s": triton_r.decode_tokens_per_sec_mean,
            "decode_speedup": decode_speedup,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nNote on decode numbers: don't read the decode-speedup column as "
        "a Stage 2 result. The eager baseline here uses eager attention "
        "for its OWN decode steps too (never optimized), while the Triton "
        "variant's decode falls back to SDPA (per the dispatch logic in "
        "flash_attention_patch.py) -- so any decode-speed difference you "
        "see reflects PyTorch's pre-existing SDPA-vs-eager advantage, not "
        "anything this stage built. TTFT is the number Stage 2 actually "
        "moved; decode wasn't touched by this kernel at all."
    )


if __name__ == "__main__":
    main()
