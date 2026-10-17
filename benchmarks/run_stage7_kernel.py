"""
Stage 7 isolated kernel benchmark: INT8 Triton kernel vs. fp16 cuBLAS.

Run this BEFORE reading the end-to-end numbers. It answers the question
those can't: is the hand-written kernel itself competitive? The kernel
is untuned (fixed block-size heuristic, no autotune -- see the kernel's
docstring), and it's competing against cuBLAS, which is heavily
optimized. An end-to-end result can't tell you whether a disappointing
speedup came from the idea (halving weight bytes) or from this specific
kernel being slower than cuBLAS -- this can.

Reports effective weight bandwidth (weight bytes read / time). Decode-
shaped calls (M=1) are memory-bound, so effective GB/s near the GPU's
peak memory bandwidth means the kernel is doing about as well as any
kernel could for that many bytes -- and since int8 reads half the bytes,
the same bandwidth means about half the time.

Usage:
    python benchmarks/run_stage7_kernel.py
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.quantization.int8 import QuantizedLinearInt8
from src.kernels.w8a16_matmul_triton import w8a16_linear

# (K, N, which Qwen2.5-1.5B projection this represents)
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
    return statistics.mean(times), (statistics.stdev(times) if n_trials > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/results/stage7_kernel.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: requires a CUDA GPU.")
        sys.exit(1)

    header = (
        f"{'layer':<14}{'K':>6}{'N':>6}{'M':>6}"
        f"{'fp16(us)':>11}{'int8(us)':>11}{'speedup':>9}"
        f"{'fp16 GB/s':>11}{'int8 GB/s':>11}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for K, N, name in SHAPES:
        torch.manual_seed(0)
        linear = torch.nn.Linear(K, N, bias=False).cuda().half()
        q = QuantizedLinearInt8.from_linear(linear)
        fp16_bytes = K * N * 2
        int8_bytes = K * N * 1

        for M in M_VALUES:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)

            fp16_ms, _ = _time_cuda(lambda: torch.nn.functional.linear(x, linear.weight), args.trials)
            int8_ms, _ = _time_cuda(lambda: w8a16_linear(x, q.weight_q, q.scale, None), args.trials)

            speedup = fp16_ms / int8_ms
            fp16_gbs = fp16_bytes / (fp16_ms / 1000) / 1e9
            int8_gbs = int8_bytes / (int8_ms / 1000) / 1e9

            print(
                f"{name:<14}{K:>6}{N:>6}{M:>6}"
                f"{fp16_ms*1000:>11.1f}{int8_ms*1000:>11.1f}{speedup:>8.2f}x"
                f"{fp16_gbs:>11.0f}{int8_gbs:>11.0f}"
            )
            results.append({
                "layer": name, "K": K, "N": N, "M": M,
                "fp16_us": fp16_ms * 1000, "int8_us": int8_ms * 1000, "speedup": speedup,
                "fp16_gbs": fp16_gbs, "int8_gbs": int8_gbs,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(
        "\nHow to read this: M=1 rows are the decode-shaped case this stage "
        "targets. speedup > 1.0x means the int8 kernel beat cuBLAS fp16 on "
        "that shape. Large-M rows (prefill-shaped) are compute-bound, where "
        "halving weight bytes helps little and an untuned kernel often "
        "loses to cuBLAS -- expected, not a failure of the idea."
    )


if __name__ == "__main__":
    main()
