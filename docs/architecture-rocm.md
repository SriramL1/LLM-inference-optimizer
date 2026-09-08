# ROCm/HIP LLM Inference Optimizer — Full SDLC Architecture

A general, code-free architecture and project plan for building a ROCm/HIP-based LLM Inference Optimizer targeting AMD Instinct/Radeon GPUs, organized across all five phases of the SDLC: **Requirements Analysis, System Design, Implementation Planning, Testing Strategy, and Deployment & Maintenance.**

This is the AMD-targeted counterpart to the original CUDA version — the underlying bottlenecks (compute vs. memory-bound phases, KV-cache pressure) are identical; what changes is the toolchain, kernel programming model, and a few platform-specific constraints.

---

## Phase 1: Requirements Analysis

### 1.1 Problem Statement
Unchanged from the CUDA version: LLM inference is bottlenecked by **compute (FLOPs)**, **memory bandwidth (HBM)**, and **memory capacity (VRAM for weights + KV cache)**. The goal is to maximize throughput and minimize latency on AMD GPU hardware, within accuracy tolerance.

### 1.2 Functional Requirements
Same as before, with one addition specific to this platform:
- Accept a pretrained transformer model and serve inference requests (batch and streaming).
- Configurable precision: FP16/BF16 baseline, optional INT8/INT4 quantization.
- Variable-length prompts/generation via dynamic batching.
- Scheduling layer for admission, batching, eviction.
- Profiling/telemetry hooks.
- **New:** target a specific AMD GPU architecture (or set of architectures) explicitly up front — e.g., CDNA-class Instinct (MI300/MI350 series) for datacenter, or RDNA-class Radeon for workstation/consumer. This decision affects kernel tuning far more than it would on a single well-standardized CUDA target.

### 1.3 Non-Functional Requirements
- **Performance targets:** throughput/latency SLAs, same as before.
- **Portability caveat (important, AMD-specific):** ROCm kernels generally need **per-architecture recompilation** — a HIP kernel built for one Instinct/RDNA generation typically isn't a drop-in binary for another. Design for this explicitly rather than assuming CUDA-style forward compatibility.
- **Reliability:** no silent numerical corruption from custom kernels or quantization.
- **Observability:** every optimization independently toggleable and measurable.

### 1.4 Constraints
- ROCm version compatibility with your ML framework (PyTorch has official ROCm builds; confirm version pinning).
- VRAM ceiling per GPU.
- Team skillset: HIP (CUDA-like C++) is the direct-control option; higher-level DSLs (Triton has an AMD backend) trade some peak performance for faster iteration.
- Documentation maturity: ROCm's docs have historically been more fragmented across sources than CUDA's; budget extra research time, and lean on ADP's engineer access to close gaps quickly.

### 1.5 Stakeholders & Success Metrics
Unchanged: tokens/sec/$, P50/P99 latency, GPU utilization %, memory footprint, post-quantization accuracy delta.

---

## Phase 2: System Design (Architecture)

### 2.1 High-Level Architecture

```
                ┌─────────────────────────────────────────┐
                │            Client / API Layer           │
                │  (gRPC/HTTP endpoint, request queueing) │
                └───────────────────┬─────────────────────┘
                                    │
                ┌───────────────────▼───────────────────────┐
                │           Scheduler / Batcher             │
                │  - Continuous (in-flight) batching        │
                │  - Request admission control              │
                │  - Priority / SLA-aware ordering          │
                └───────────────────┬───────────────────────┘
                                    │
                ┌───────────────────▼───────────────────────┐
                │        Execution Engine (per GPU)         │
                │  - Model graph executor                   │
                │  - HIP Graph capture/replay for decode    │
                │  - Kernel dispatch (fused attention, MLP) │
                └───────────────────┬───────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐        ┌─────────▼─────────┐       ┌─────────▼─────────┐
│  Memory Manager│        │  Kernel Library   │       │  Parallelism Layer│
│ -Paged KV cache│        │ - Fused attention │       │ -Tensor parallel  │
│ -Cache eviction│        │  (HIP / Composable│       │ -Pipeline parallel│
│ -Quant KV cache│        │    Kernel-based)  │       │ -RCCL collectives │
└────────────────┘        │ - Fused LayerNorm │       └───────────────────┘
                          │ - Quant matmul    │
                          └───────────────────┘
                                    │
                ┌───────────────────▼───────────────────────┐
                │        Profiling & Telemetry Layer        │
                │  rocprof / ROCm Compute Profiler hooks    │
                └───────────────────────────────────────────┘
```

### 2.2 Component Responsibilities & Platform Mapping

**Client/API Layer** — unchanged; framework-agnostic.

**Scheduler/Batcher** — unchanged conceptually (continuous batching, admission control, speculative-decoding support). No AMD-specific dependency here; this logic sits above the hardware layer.

**Execution Engine**
- Same prefill (compute-bound) / decode (memory-bound) split as the CUDA version.
- **HIP Graphs** are the ROCm equivalent of CUDA Graphs — capture-and-replay a fixed sequence of kernel launches for the decode step to cut launch overhead. Support and performance characteristics can lag CUDA Graphs on some ROCm versions, so validate this specifically rather than assuming parity.

**Memory Manager** — same paged-KV-cache design as before; this is a scheduling/data-structure problem, not a hardware-specific one, so it ports conceptually unchanged.

**Kernel Library**
- Written in **HIP** (near-1:1 syntax mapping from CUDA C++: `hipMalloc`/`hipMemcpy`/`hipLaunchKernelGGL` in place of the `cuda*` equivalents).
- For fused attention specifically, look at **Composable Kernel (CK)**, AMD's template library purpose-built for fused attention/GEMM on CDNA hardware — it's the closest analog to CUTLASS/FlashAttention's role in the CUDA stack.
- **hipBLAS/rocBLAS** replace cuBLAS for GEMM primitives; **MIOpen** replaces cuDNN for common deep-learning primitives.
- If you want faster kernel iteration than raw HIP, **Triton has an AMD/ROCm backend** — same trade-off as on NVIDIA (some peak-perf cost for much faster development).

**Parallelism Layer**
- Tensor/pipeline parallelism concepts are unchanged.
- **RCCL** (ROCm Collective Communication Library) replaces NCCL for multi-GPU all-reduce/broadcast operations.

**Profiling & Telemetry**
- **rocprof** and the newer **ROCm Compute Profiler / Omnitrace** replace Nsight Systems/Compute. Same roofline-style compute-vs-memory-bound analysis is available, though tooling maturity and UI polish vary more across ROCm versions than the Nsight suite does — worth a short spike to confirm your specific ROCm version's profiler output is trustworthy before relying on it for Phase 4 gating.

### 2.3 Key Design Decisions & Trade-offs (Updated for ROCm)

| Decision | Option A | Option B | Trade-off |
|---|---|---|---|
| Kernel authoring | Raw HIP C++ | Triton (ROCm backend) | HIP = max control, closer to hardware; Triton = faster iteration, backend maturity varies by ROCm version. |
| Attention kernel base | Hand-written HIP | Composable Kernel (CK) library | CK gives you a tuned, hardware-aware starting point; hand-written gives full control but more debugging surface. |
| KV cache layout | Contiguous per-sequence | Paged (block-based) | Same trade-off as CUDA version — platform-agnostic. |
| Quantization | Weight-only | Weight+activation | Same trade-off as CUDA version; verify kernel-level INT8/INT4 support maturity for your specific target architecture. |
| Multi-arch support | Single target arch (e.g., MI300 only) | Multiple arches (MI300 + RDNA) | Multiple arches means maintaining separate compiled kernel variants and testing matrices — budget for this explicitly (see 1.3). |

### 2.4 Data Flow (Single Request Lifecycle)
Identical to the CUDA version — the lifecycle (admit → allocate KV blocks → prefill → fold into decode batch → per-step decode with cache append → stream tokens → free blocks) is a scheduling/architecture concept independent of the underlying GPU vendor.

---

## Phase 3: Implementation Planning

### 3.1 Suggested Build Order
Same sequencing logic as the CUDA plan, with tool substitutions:

1. **Baseline harness** — naive FP16 inference on ROCm-enabled PyTorch, profiling wired up via rocprof from day one.
2. **Fused attention kernel** — start from Composable Kernel's attention templates rather than writing fully from scratch; adapt/extend rather than reinvent.
3. **Fused norm + activation kernels** — HIP, following the same fusion logic as the CUDA plan.
4. **Paged KV-cache manager** — platform-agnostic; port design directly.
5. **Continuous batching scheduler** — platform-agnostic; port design directly.
6. **HIP Graphs for decode** — apply once your kernel set is stable; test this stage carefully given more variable maturity across ROCm releases.
7. **Quantization** — weight-only first; verify rocBLAS/CK support for your target precision before committing to a kernel design.
8. **Speculative decoding** — same approach as CUDA version, hardware-agnostic at the scheduling level.
9. **Multi-GPU parallelism** — tensor/pipeline parallel using RCCL in place of NCCL.

### 3.2 Tooling & Environment
- **AMD Developer Cloud** for on-demand Instinct GPU access (removes the "no local GPU" blocker).
- ROCm stack (HIP, rocBLAS, MIOpen, RCCL) matched to your target architecture(s).
- Composable Kernel for attention/GEMM kernel templates.
- Triton (ROCm backend) if choosing DSL-based kernels.
- PyTorch with ROCm build as host framework.
- rocprof / ROCm Compute Profiler / Omnitrace for profiling.
- **ADP resources specifically:** AMD engineer access and private community channels for unblocking ROCm-specific issues faster than public forums; ROCm.ai/Hyperloom as both a reference benchmark and a potential development accelerant (see below).

### 3.3 Using ROCm.ai / Hyperloom in the Build
Two concrete uses, per your ADP membership:
- **Baseline comparison:** run Hyperloom's automated optimization on your model early, before you hand-build anything. Record what it fuses, how it schedules, and what speedup it achieves — this becomes both a target to beat and a source of design ideas.
- **Documentation shortcut:** ROCm.ai's integration with coding assistants (including Claude) can reduce time spent hunting fragmented ROCm docs. Use it to accelerate lookup, not to replace understanding the memory-hierarchy reasoning your fused kernels depend on — the latter is what Phase 4 correctness testing will expose gaps in if skipped.

### 3.4 Risk Register (Updated)
| Risk | Mitigation |
|---|---|
| Custom HIP kernel numerically diverges from reference | Golden-output diff testing against PyTorch reference at every stage |
| HIP Graph capture behaves differently than expected on your ROCm version | Spike-test HIP Graphs early and in isolation before depending on them |
| Kernel tuned for one AMD architecture underperforms/fails on another | Explicit per-architecture build matrix; do not assume portability |
| ROCm documentation gaps slow debugging | Use ADP engineer access and community channels proactively, don't rely solely on public docs |
| Quantization kernel support less mature for target precision | Verify rocBLAS/CK support for your exact quant scheme before committing design |

---

## Phase 4: Testing Strategy

### 4.1 Correctness Testing
Same structure as CUDA version — unit-level kernel diffs against PyTorch reference, integration-level full-forward-pass diffs, regression suite on every change. Tooling: use `rocprof`/ROCm Compute Profiler for the performance side; correctness testing itself is framework-level (PyTorch) and platform-agnostic.

### 4.2 Performance Testing
- **Microbenchmarks:** per-kernel latency and achieved bandwidth vs. theoretical peak, using ROCm Compute Profiler's roofline-style output.
- **End-to-end benchmarks:** same throughput/latency-percentile methodology as the CUDA version.
- **Cross-architecture testing (AMD-specific addition):** if supporting multiple AMD GPU generations, re-run the full performance suite per architecture — do not assume results transfer.

### 4.3 Accuracy Testing (Quantization-Specific)
Unchanged methodology: perplexity + task-based eval, pre/post quantization, defined acceptance threshold.

### 4.4 Reliability/Chaos Testing
Unchanged: memory-pressure tests, multi-GPU failure injection (using RCCL in place of NCCL for the collective-failure scenario).

---

## Phase 5: Deployment & Maintenance

### 5.1 Deployment Architecture
- Containerize with a specific ROCm version pinned (ROCm version/kernel-driver compatibility matters as much as CUDA/driver pinning does).
- Canary deployment, feature-flagged optimizations — unchanged from CUDA version.
- **Architecture-aware routing:** if serving on a mix of AMD GPU generations, your deployment layer needs to route requests to the correctly-compiled kernel variant for the hardware present — an extra piece of routing logic not present in a single-architecture NVIDIA deployment.

### 5.2 Monitoring in Production
Unchanged metric set; ingest from `rocprof`/Omnitrace instead of Nsight-based exporters.

### 5.3 Maintenance Plan
- **ROCm version upgrades:** re-run full regression suite before adopting a new ROCm release — historically more impactful on kernel behavior than equivalent CUDA upgrades, given the faster pace of ROCm's evolution (e.g., recent jumps like ROCm 7.x brought significant performance and compatibility changes).
- **New AMD architecture support:** treat as a full mini-Phase-3 cycle (re-tune, re-benchmark, re-validate) rather than assuming portability — this is the single biggest recurring cost difference vs. the CUDA version of this project.
- **Model updates:** re-validate fused kernels against new tensor shapes/strides, same as CUDA version.
- **Continuous benchmarking:** keep the baseline harness alive permanently; also keep a periodic Hyperloom re-run as an external sanity check that your hand-tuned pipeline still compares favorably to AMD's own automated optimizer as ROCm evolves.

---

## Summary

The bottleneck analysis is identical to the CUDA version — prefill is compute-bound, decode is memory-bound, KV cache dominates memory pressure — because that's a property of the transformer architecture, not the GPU vendor. What changes on ROCm is: HIP instead of CUDA C++ for kernels, Composable Kernel instead of CUTLASS/FlashAttention source as your fused-attention reference, rocprof/Omnitrace instead of Nsight for profiling, RCCL instead of NCCL for multi-GPU, and — most structurally important — an explicit, budgeted assumption that **kernels need per-architecture recompilation** rather than being written once and running everywhere. Your AMD AI Developer Program membership offsets the biggest practical risk here (documentation fragmentation and debugging friction) via direct engineer access, and Hyperloom gives you both a benchmark target and a development accelerant you don't have an equivalent for on the NVIDIA side.
