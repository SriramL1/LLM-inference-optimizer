# Stage 1: Baseline Inference Harness

## Goal

Establish an unoptimized, obviously-correct reference implementation and a
profiling harness, so every later optimization stage has something concrete
to measure itself against.

## What "unoptimized" means here

- `attn_implementation="eager"` — no PyTorch SDPA, no FlashAttention. Fused
  attention is Stage 2; baking it into the baseline would make Stage 2's
  speedup invisible.
- No batching tricks, no CUDA graphs, no quantization, no custom kernels.
  Just HuggingFace's default forward pass plus its built-in KV-cache.
- Prefill and decode are separate, explicit function calls
  (`engine.prefill`, `engine.decode_step`) rather than a single
  `model.generate()` call, so the profiler can time each phase
  independently with CUDA events.

## What's measured

| Metric | What it captures | Why it matters |
|---|---|---|
| TTFT (ms) | Time for the full prompt's forward pass (prefill) | Prefill is compute-bound — this is what Stage 2 (fused attention) targets |
| Decode tokens/sec | Steady-state throughput once the KV-cache is warm | Decode is memory-bandwidth-bound — this is what Stages 4-6 (paged KV-cache, continuous batching, CUDA graphs) target |
| Peak memory (MB) | `torch.cuda.max_memory_allocated` | Baseline for judging Stage 4 (paged KV-cache) and Stage 7 (quantization) memory savings |
| Per-token latency | Individual decode step timings | Reveals variance/tail latency, not just the average |

Timing uses `torch.cuda.Event`, not `time.time()`, so it captures GPU
kernel time rather than CPU-side dispatch overhead.

## Model choice

Default: `Qwen/Qwen2.5-1.5B-Instruct` — small enough to run comfortably on
a 12GB RTX 4070, but uses the same architectural building blocks
(RMSNorm, RoPE, grouped-query attention) as the larger models these
techniques target in production. Swappable via `--model`.

## Running it

```bash
pip install -r requirements.txt
python benchmarks/run_baseline.py
```

Results print as a summary table and save to
`benchmarks/results/stage1_baseline.json`. Sweep batch size / generation
length with `--batch-sizes` and `--gen-lengths`.

## Correctness check

`tests/test_baseline_engine.py` asserts the manual prefill/decode loop
produces identical greedy-decoded tokens to `model.generate()`. Run with:

```bash
pytest tests/test_baseline_engine.py -v
```

Every later stage should pass the equivalent of this test — optimizations
are allowed to change speed and memory, never the output tokens (for
greedy decoding).

## Hardware note

Baseline numbers here are from a consumer RTX 4070 (12GB). Later stages
ported to ROCm/HIP (via the AMD AI Developer Program's MI300X cloud
access) will be benchmarked separately — MI300X is a datacenter part, so
cross-platform numbers should be read as "does the optimization technique
transfer," not "which GPU is faster."

## Known limitations (intentional, for now)

- No batching efficiency beyond repeating the same prompt — real
  variable-length batching arrives with continuous batching (Stage 5).
- No PagedAttention-style memory management — the KV-cache uses whatever
  HuggingFace allocates by default (Stage 4 replaces this).
- Single-GPU only (Stage 9 covers multi-GPU).
