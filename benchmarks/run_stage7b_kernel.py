"""
Stage 7b isolated kernel benchmark: INT4 vs. INT8 vs. fp16 cuBLAS.

Same purpose as run_stage7_kernel.py, one more column. Effective GB/s is
weight bytes actually stored (including INT4 group scales) divided by
time. Run this before reading the end-to-end numbers.

Usage:
    python benchmarks/run_stage7b_kernel.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.quantization.int8 import QuantizedLinearInt8
from src.quantization.int4 import QuantizedLinearInt4
from src.kernels.w8a16_matmul_triton import w8a16_linear
from src.kernels.w4a16_matmul_triton import w4a16_linear

SHAPES = [
    (1536, 1536, "q/o proj"),
    (1536, 256, "k/v proj"),
    (1536, 8960, "gate/up proj"),
    (8960, 1536, "down proj"),
]
M_VALUES = [1, 8, 128, 1024]


def _time_cuda(fn, n_trials, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.mean(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/stage7b_kernel.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: requires a CUDA GPU.")
        sys.exit(1)

    header = (
        f"{'layer':<14}{'K':>6}{'N':>6}{'M':>6}"
        f"{'fp16(us)':>10}{'int8(us)':>10}{'int4(us)':>10}"
        f"{'i8 spd':>8}{'i4 spd':>8}{'i4 GB/s':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for K, N, name in SHAPES:
        torch.manual_seed(0)
        linear = torch.nn.Linear(K, N, bias=False).cuda().half()
        q8 = QuantizedLinearInt8.from_linear(linear)
        q4 = QuantizedLinearInt4.from_linear(linear)
        int4_bytes = q4.num_bytes()

        for M in M_VALUES:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)

            fp16_ms = _time_cuda(lambda: torch.nn.functional.linear(x, linear.weight), args.trials)
            int8_ms = _time_cuda(lambda: w8a16_linear(x, q8.weight_q, q8.scale, None), args.trials)
            int4_ms = _time_cuda(lambda: w4a16_linear(x, q4.weight_packed, q4.scales, None), args.trials)

            i4_gbs = int4_bytes / (int4_ms / 1000) / 1e9
            print(
                f"{name:<14}{K:>6}{N:>6}{M:>6}"
                f"{fp16_ms*1000:>10.1f}{int8_ms*1000:>10.1f}{int4_ms*1000:>10.1f}"
                f"{fp16_ms/int8_ms:>7.2f}x{fp16_ms/int4_ms:>7.2f}x{i4_gbs:>9.0f}"
            )
            results.append({
                "layer": name, "K": K, "N": N, "M": M,
                "fp16_us": fp16_ms * 1000, "int8_us": int8_ms * 1000, "int4_us": int4_ms * 1000,
                "int8_speedup": fp16_ms / int8_ms, "int4_speedup": fp16_ms / int4_ms,
                "int4_gbs": i4_gbs,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nHow to read this: speedups are vs. cuBLAS fp16 (>1.0x = faster). "
        "M=1 is the decode-shaped, memory-bound case. INT4 does extra "
        "per-element work (nibble unpack + per-group scale), so it can lose "
        "to INT8 even though it reads half the bytes -- if it does, the "
        "kernel's arithmetic, not the bandwidth, is the bottleneck."
    )


if __name__ == "__main__":
    main()
