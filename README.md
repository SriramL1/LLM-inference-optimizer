# LLM Inference Optimizer

A from-scratch project exploring GPU-level optimization of large language model (LLM) inference, targeting both **NVIDIA (CUDA)** and **AMD (ROCm/HIP)** platforms.

The project follows the full Software Development Life Cycle (SDLC) — requirements, design, implementation, testing, and deployment — applied to a real systems-engineering problem: making transformer inference faster and cheaper on GPU hardware.

## Why this project

LLM inference has two very different performance regimes:
- **Prefill** (processing the prompt): compute-bound.
- **Decode** (generating tokens one at a time): memory-bandwidth-bound.

Nearly every optimization technique here — kernel fusion, paged KV-cache, continuous batching, quantization, speculative decoding — exists to attack one of those two bottlenecks. This repo documents and implements those techniques incrementally, with every optimization independently measured against a fixed baseline.

## Platforms

| Platform | Kernel language | Key libraries |
|---|---|---|
| NVIDIA (CUDA) | CUDA C++ / Triton | cuBLAS, cuDNN, CUTLASS, NCCL |
| AMD (ROCm) | HIP / Triton (ROCm backend) | rocBLAS, MIOpen, Composable Kernel, RCCL |

## Roadmap (build order)

1. Baseline inference harness + profiling instrumentation
2. Fused attention kernel (FlashAttention-style)
3. Fused LayerNorm/RMSNorm + activation kernels
4. Paged KV-cache manager
5. Continuous batching scheduler
6. CUDA Graphs / HIP Graphs for decode
7. Weight quantization (INT8/INT4)
8. Speculative decoding
9. Multi-GPU parallelism (tensor/pipeline)

Each stage is implemented and benchmarked independently against the Stage-1 baseline before moving to the next.

## Repository structure

```
.
├── docs/
│   ├── architecture-cuda.md      # Full SDLC architecture doc (CUDA target)
│   └── architecture-rocm.md      # Full SDLC architecture doc (ROCm/HIP target)
├── src/
│   ├── kernels/                  # Custom CUDA/HIP kernels
│   ├── scheduler/                # Batching + admission control
│   ├── memory/                   # Paged KV-cache manager
│   └── engine/                   # Execution engine / graph capture
├── benchmarks/                   # Microbenchmarks + end-to-end throughput/latency tests
├── tests/                        # Correctness + regression tests
└── README.md
```

## Status

🚧 Early stage — baseline harness and architecture design in progress.

## Getting started

_TBD once Stage 1 (baseline harness) lands — will include setup instructions for both CUDA and ROCm environments._

## References

- FlashAttention (Dao et al.)
- vLLM / PagedAttention (Kwon et al.)
- NVIDIA CUTLASS / TensorRT-LLM
- AMD Composable Kernel / ROCm documentation

## License

TBD
