# LLM Inference Optimizer

A from-scratch project exploring GPU-level optimization of large language model (LLM) inference, targeting both **NVIDIA (CUDA)** and **AMD (ROCm/HIP)** platforms.

The project follows the full Software Development Life Cycle (SDLC) — requirements, design, implementation, testing, and deployment — applied to a real systems-engineering problem: making transformer inference faster and cheaper on GPU hardware.

## Why this project

LLM inference has two very different performance regimes:

- **Prefill** (processing the prompt): compute-bound.
- **Decode** (generating tokens one at a time): memory-bandwidth-bound.

Nearly every optimization technique here — kernel fusion, paged KV-cache, continuous batching, quantization, speculative decoding — exists to attack one of those two bottlenecks. This repo documents and implements those techniques incrementally, with every optimization independently measured against a fixed baseline.

## Platforms

| Platform      | Kernel language             | Key libraries                            |
| ------------- | --------------------------- | -----------------------------------------|
| NVIDIA (CUDA) | CUDA C++ / Triton           | cuBLAS, cuDNN, CUTLASS, NCCL             |
| AMD (ROCm)    | HIP / Triton (ROCm backend) | rocBLAS, MIOpen, Composable Kernel, RCCL |

CUDA development runs locally (RTX 4060). ROCm ports will run on AMD Instinct MI300X via the AMD AI Developer Program's cloud credits, for stages where portability is the point.

## Roadmap (build order)

1. ✅ Baseline inference harness + profiling instrumentation
2. ✅ Fused attention kernel (FlashAttention-style, Triton)
3. ⬜ Fused LayerNorm/RMSNorm + activation kernels
4. ⬜ Paged KV-cache manager
5. ⬜ Continuous batching scheduler
6. ⬜ CUDA Graphs / HIP Graphs for decode
7. ⬜ Weight quantization (INT8/INT4)
8. ⬜ Speculative decoding
9. ⬜ Multi-GPU parallelism (tensor/pipeline)

Each stage is implemented and benchmarked independently against the Stage 1 baseline before moving to the next.

## Results so far

Hardware: NVIDIA RTX 4060 (8GB), Qwen2.5-1.5B-Instruct, fp16.

**Stage 1 (baseline, eager attention)** — decode scales near-linearly with batch size (memory-bandwidth-bound, as expected): ~47 tok/s at batch 1 → ~384 tok/s at batch 8. Full methodology and mean±std across multiple trials in [`docs/stage1-baseline.md`](stage1-baseline/docs/stage1-baseline.md).

**Stage 2 (Triton fused attention)**:
- Isolated kernel vs. PyTorch SDPA: matches within noise (0.98x–1.00x) at 2048–4096 tokens, ~14x faster than a naive unfused reference at 4096 tokens.
- Wired into the real model via HuggingFace's `AttentionInterface`, with correctness validated both by direct logit comparison and greedy-decoding agreement against SDPA on real prompts.
- **End-to-end TTFT (time to first token): 1.46x faster at 2048-token prompts**, scaling up from 1.04x at 128 tokens — growing with sequence length as predicted, since attention's share of prefill compute grows with context length.
- Full writeup, including three real integration bugs hit and fixed along the way (a transformers naming collision, a flawed correctness-test baseline, and an over-strict floating-point tolerance), in [`docs/stage2-fused-attention.md`](stage1-baseline/docs/stage2-fused-attention.md) and [`docs/stage2b-integration.md`](stage1-baseline/docs/stage2b-integration.md).

## Repository structure

```
.
├── docs/
│   ├── architecture-cuda.md          # Full SDLC architecture doc (CUDA target)
│   └── architecture-rocm.md          # Full SDLC architecture doc (ROCm/HIP target)
└── stage1-baseline/                  # Stages 1-2 (to be flattened to repo root in a later cleanup pass)
    ├── docs/
    │   ├── stage1-baseline.md        # Stage 1 methodology
    │   ├── stage2-fused-attention.md # Stage 2 kernel methodology
    │   └── stage2b-integration.md    # Stage 2 end-to-end integration notes
    ├── src/
    │   ├── kernels/                  # Custom Triton/CUDA kernels
    │   └── engine/                   # Model loading, inference loop, attention dispatch, profiling
    ├── benchmarks/                   # Microbenchmarks + end-to-end throughput/latency tests
    ├── tests/                        # Correctness + regression tests
    └── requirements.txt
```

## Getting started

Requires a CUDA-capable NVIDIA GPU (developed against an RTX 4060, 8GB) or an AMD GPU via ROCm (later stages).

```bash
cd stage1-baseline
pip install -r requirements.txt

# CUDA-enabled torch (pip's default torch wheel is CPU-only on Windows):
pip install torch --index-url https://download.pytorch.org/whl/cu126  # match your driver's CUDA version

# Triton (Stage 2+):
# Linux/WSL2:      pip install triton
# Windows (native): pip install -U "triton-windows<3.7"  — also requires MSVC Build Tools
#                    ("Desktop development with C++" workload)
```

Run the Stage 1 baseline:
```bash
python benchmarks/run_baseline.py
```

Run Stage 2 (kernel correctness + isolated benchmark + end-to-end benchmark):
```bash
pytest tests/test_flash_attention_kernel.py -v
python benchmarks/run_stage2_attention.py

pytest tests/test_e2e_attention_patch.py -v
python benchmarks/run_stage2_e2e.py
```

## References

- FlashAttention (Dao et al.)
- vLLM / PagedAttention (Kwon et al.)
- NVIDIA CUTLASS / TensorRT-LLM
- AMD Composable Kernel / ROCm documentation
- Triton (OpenAI) / [triton-windows](https://github.com/triton-lang/triton-windows)

## License

TBD