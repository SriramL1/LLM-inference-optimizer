# Stage 2: Fused Attention Kernel (Triton)

## Goal

Replace the Stage 1 eager attention path (materializes the full
`seq_len x seq_len` attention matrix, three separate memory-bound ops:
matmul → softmax → matmul) with a single fused Triton kernel that never
writes the intermediate attention matrix to global memory. This targets
**prefill** specifically -- the compute-bound phase Stage 1's numbers
identified as the place kernel fusion pays off.

## Why this should be faster

Eager attention does three separate GPU kernel launches, each reading and
writing to global memory (HBM) in between:
1. `QK^T` → write `(seq_len, seq_len)` scores to HBM
2. `softmax(scores)` → read scores, write probabilities back to HBM
3. `probs @ V` → read probabilities back from HBM

For long sequences, step 2 in particular is expensive purely from memory
traffic, not compute -- softmax itself is cheap, but reading and writing
an `O(seq_len^2)` matrix isn't.

The fused kernel (FlashAttention algorithm) tiles Q, K, V into blocks,
computes attention block-by-block entirely in fast on-chip memory (SRAM /
registers), and uses "online softmax" (a running max + running sum,
rescaled as new blocks arrive) to avoid ever needing the full row of
scores at once. The `(seq_len, seq_len)` matrix never touches HBM.

## Scope: prefill only

This kernel assumes self-attention -- query, key, and value all have the
same sequence length. That's prefill. Decode (one new query token
attending over a growing KV-cache) is a different shape problem entirely
and stays on the Stage 1 eager path for now; it's memory-bandwidth-bound
in a different way that Stage 4 (paged KV-cache) and Stage 6 (CUDA
graphs) target.

## Correctness strategy

Three-way cross-check, not just "close enough to one reference":
1. **Naive PyTorch reference** (`naive_attention_reference`) -- unfused,
   obviously-correct-by-construction. The ground truth.
2. **PyTorch SDPA** (`scaled_dot_product_attention`) -- an independent
   optimized implementation. If the Triton kernel and the naive reference
   agreed on a shared bug, this would likely catch it.
3. Shapes tested include **non-multiple-of-block-size sequence lengths**
   (e.g. 65) specifically to exercise the masking logic at tile boundaries
   -- the most common place off-by-one bugs hide in tiled kernels.

Run with:
```bash
pytest tests/test_flash_attention_kernel.py -v
```

## Benchmark methodology

`benchmarks/run_stage2_attention.py` sweeps sequence length (the axis
that matters for prefill) rather than batch size (Stage 1's axis), and
reports the attention op in isolation -- not full end-to-end model
inference. This isolates the kernel's actual behavior from everything
else in the model (MLP layers, layernorm, etc.), which matters because
Amdahl's Law means even a huge attention speedup produces a smaller
end-to-end speedup once the rest of the model's cost is included.

Two comparison points are reported, not one:
- **vs naive** -- the easy bar. Any correct fused kernel beats this by a
  wide margin at long sequence lengths.
- **vs SDPA** -- the real bar. This is PyTorch's own production
  flash-attention backend (cuDNN/FlashAttention-2 under the hood on most
  setups). Matching or beating this is what actually demonstrates the
  kernel is competitive, not just "faster than doing it badly."

## Known limitations (intentional, for now)

- GQA is handled by `repeat_interleave`-ing K/V heads before the kernel
  call, which duplicates memory rather than broadcasting KV heads inside
  the kernel. Correct, but leaves performance on the table for
  high-group-size GQA models -- a candidate for a later optimization pass,
  not Stage 2's job.
- Fixed block sizes (`BLOCK_M=64, BLOCK_N=64`), no autotuning across GPU
  architectures or head dimensions yet.
- `head_dim` must be a power of 2 (true for essentially every current LLM
  architecture, so not a practical limitation, but worth noting as an
  explicit assumption).
- No end-to-end integration into the actual model's forward pass yet --
  this stage validates the kernel in isolation first. Wiring it into
  `BaselineEngine` (replacing HF's eager attention call inside the model)
  is a natural Stage 2b follow-up once the kernel is validated on your
  hardware.

## Windows/Triton note

Official `pip install triton` wheels are Linux-only. This project uses
the community-maintained `triton-windows` fork
(https://github.com/triton-lang/triton-windows), which requires MSVC
Build Tools for on-the-fly kernel compilation. See the main README for
setup steps.
