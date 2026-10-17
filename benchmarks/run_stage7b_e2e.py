"""
Stage 7b end-to-end benchmark: fp16 vs. INT8 vs. INT4 weights.

Reuses the measurement code from run_stage7_e2e.py (same HF-path and
graphed-path methodology and the same two precautions: Triton JIT before
graph capture, max_context_len=512). Each precision gets a FRESH model
load, since a model quantized in place can't be quantized again at a
different precision. Also reports an accuracy column -- logit cosine
similarity and top-1 agreement vs. fp16 on a fixed prompt -- because INT4
trades noticeably more accuracy for its extra compression, and a speed
table without it would be misleading.

Usage:
    python benchmarks/run_stage7b_e2e.py
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.engine.model_loader import load_model
from src.quantization.int8 import quantize_model_int8
from src.quantization.int4 import quantize_model_int4
from run_stage7_e2e import run_phase

QUALITY_PROMPT = (
    "The history of computing spans many decades, from early mechanical "
    "calculators to modern electronic computers. Charles Babbage designed "
    "the Analytical Engine in the 19th century, though it was never fully "
    "built in his lifetime. Ada Lovelace wrote what is considered the "
    "first algorithm intended for machine execution."
)


def quality_logits(loaded):
    ids = loaded.tokenizer(QUALITY_PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        return loaded.model(input_ids=ids).logits.float().cpu()


def free(loaded):
    del loaded
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="benchmarks/results/stage7b_e2e.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: requires a CUDA GPU.")
        sys.exit(1)

    phases, quality, stats_by = {}, {}, {}
    ref_logits = None

    for precision, quantize in (("fp16", None), ("int8", quantize_model_int8), ("int4", quantize_model_int4)):
        print(f"\nLoading {args.model} [sdpa, fp16] for {precision}...")
        loaded = load_model(args.model, device="cuda", attn_implementation="sdpa")
        if quantize is not None:
            stats = quantize(loaded.model)
            stats_by[precision] = stats
            print(f"Quantized {stats['layers_quantized']} layers "
                  f"({stats['bytes_before']/1024**2:.0f} MB -> {stats['bytes_after']/1024**2:.0f} MB)")
            if stats["layers_quantized"] == 0:
                print("WARNING: nothing was quantized -- numbers below are really fp16.")

        logits = quality_logits(loaded)
        if precision == "fp16":
            ref_logits = logits
            quality[precision] = {"cosine": 1.0, "top1_agreement": 1.0}
        else:
            cos = torch.nn.functional.cosine_similarity(ref_logits, logits, dim=-1).mean().item()
            agree = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
            quality[precision] = {"cosine": cos, "top1_agreement": agree}

        phases[precision] = run_phase(f"{precision} weights", loaded, args)
        free(loaded)

    base = phases["fp16"]
    print("\n=== Comparison (vs fp16) ===")
    header = f"{'':<6}{'layer MB':>10}{'size':>7}{'HF b=1':>9}{'HF b=8':>9}{'graphed':>10}{'cosine':>9}{'top-1':>8}"
    print(header)
    print("-" * len(header))
    for p in ("fp16", "int8", "int4"):
        ph = phases[p]
        row = (
            f"{p:<6}{ph['layer_weight_mb']:>10.0f}"
            f"{base['layer_weight_mb']/ph['layer_weight_mb']:>6.2f}x"
        )
        for b in args.batch_sizes:
            row += f"{ph['hf'][b]['decode_tok_s']/base['hf'][b]['decode_tok_s']:>8.2f}x"
        row += f"{ph['graphed_decode_tok_s']/base['graphed_decode_tok_s']:>9.2f}x"
        row += f"{quality[p]['cosine']:>9.4f}{quality[p]['top1_agreement']:>8.1%}"
        print(row)
    print("\nAbsolute graphed decode tok/s: " + ", ".join(
        f"{p}={phases[p]['graphed_decode_tok_s']:.1f}" for p in ("fp16", "int8", "int4")))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"phases": phases, "quality": quality, "quantize_stats": stats_by}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
