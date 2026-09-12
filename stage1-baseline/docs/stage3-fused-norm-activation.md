# Stage 3: Fused RMSNorm + SwiGLU Activation Kernels

## Goal

Fuse two small, frequently-called ops that PyTorch currently runs as
several separate kernel launches each:

- **RMSNorm** (runs twice per transformer layer -- before attention,
  before the MLP): unfused, it's `pow → mean → rsqrt → mul → mul`, five
  separate CUDA kernel launches with memory round trips between each.
- **SwiGLU activation** (`silu(gate) * up`, once per MLP): unfused, it's
  `silu` writing a full intermediate tensor, then a separate multiply
  reading it back.

Neither op does much actual compute -- they're memory-bound, and the
cost is dominated by kernel launch overhead and redundant memory
traffic, not arithmetic. Fusing each into a single kernel removes both.

## Honest expectation-setting

Unlike Stage 2 (attention), these ops are a small fraction of a
transformer layer's total FLOPs -- the matmuls in attention and the MLP
projections dominate. **Don't expect an attention-sized speedup here.**
The realistic win is:
- Fewer kernel launches (dozens per forward pass across a full model,
  each with fixed overhead)
- Less memory traffic (no intermediate tensors written and re-read)
- More visible at smaller batch/sequence sizes, where launch overhead is
  a larger fraction of total time

## Implementation notes

- **RMSNorm**: one Triton program per row (one token's hidden vector).
  The whole hidden dimension is normalized together, so it has to fit in
  one block -- true for any current LLM hidden size (well under Triton's
  practical block-size ceiling). Reference matches HF's actual formula
  exactly, including the fp32-compute-then-cast-back order, which
  matters for numerical agreement, not just the math.
- **SwiGLU activation**: purely elementwise, flat 1D blocking regardless
  of logical tensor shape.

## Integration approach: different from Stage 2, and why

Stage 2 used transformers' `AttentionInterface` -- a proper, documented,
supported extension point. There's no equivalent for norm/activation
implementations, so Stage 3 patches module *instances* directly:
`patch_model_with_fused_kernels()` walks the loaded model, matches
modules by class name (`Qwen2RMSNorm`, `Qwen2MLP`, and the equivalent
Llama/Mistral names, since they share the same structure), and rebinds
`.forward` to a version calling the fused kernel.

This is less robust than Stage 2's approach by construction: an
unrecognized architecture silently patches nothing rather than raising
an error. `patch_model_with_fused_kernels()` returns a count of what it
patched specifically so callers can check it actually did something
(`tests/test_e2e_stage3.py`'s first test exists purely to catch this
failure mode before trusting anything downstream).

## Correctness strategy

Two layers, following the Stage 2 playbook:
1. **Isolated kernel tests** (`test_fused_norm_activation_kernels.py`)
   against naive PyTorch references, including non-power-of-2 sizes to
   exercise masking.
2. **End-to-end tests** (`test_e2e_stage3.py`) on the same model instance,
   patched in-place, real prompt -- logits comparison (loose magnitude
   bound + strict argmax-agreement check, per the lesson learned in
   Stage 2b about full-tensor `allclose` being too strict) and generation
   match.

Run both:
```bash
pytest tests/test_fused_norm_activation_kernels.py tests/test_e2e_stage3.py -v
```

## Benchmarks

Isolated kernel comparison, sweeping row count:
```bash
python benchmarks/run_stage3_kernels.py
```

End-to-end, single model instance timed before/after patching:
```bash
python benchmarks/run_stage3_e2e.py
```

One thing that differs from Stage 2's end-to-end benchmark: **decode
should also improve here**, not just TTFT. RMSNorm and the MLP
activation run during every decode step too, not just prefill -- unlike
Stage 2's kernel, which only touched the prefill path.

Pass `--attn-implementation triton_attn` to measure Stage 2 and Stage 3
stacked together rather than Stage 3 in isolation.

## A real bug found along the way: "eager" attention itself is unstable here

While debugging what looked like a Stage 3 correctness failure (patched
model logits came back 100% NaN), extensive bisection -- isolated kernel
tests, testing the kernel directly on real captured activations,
comparing exact discrepancy magnitude (down to confirming it was a
single float16 ULP, i.e. as close as two independent computations can
possibly be), testing the patching mechanism with the exact original
formula, testing pure passthrough replacement, testing module
replacement instead of forward-monkeypatching -- eventually proved
**none of that was the cause**. The actual finding: a completely
unpatched, vanilla model, loaded with `attn_implementation="eager"`,
produces NaN logits on a plain forward pass in this environment (torch
2.14+cu126, transformers 5.16.1, RTX 4060). `"sdpa"` on the identical
model, prompt, and hardware is clean.

This retroactively explains an earlier anomaly from Stage 2b (a
generate() comparison against an "eager" reference produced suspicious
repeated-zero output) that was, at the time, attributed to fp16
tie-breaking noise on random input -- `torch.argmax` on an all-NaN row
conventionally returns index 0, which matches that symptom exactly
better than the tie-breaking explanation does.

**Practical takeaway**: `"eager"` is treated in this repo as an
intentionally-unoptimized reference for measuring relative *speed*
(Stage 1's whole premise), not as a source of correct numerical values
to test *against*. All Stage 3 correctness tests and this stage's
end-to-end benchmark use `"sdpa"` as the reference instead, for exactly
the reason Stage 2b already established. Stage 1's own baseline timings
remain valid (NaN propagation doesn't change matmul wall-clock time),
but the actual generated text produced during Stage 1 benchmarking was
very likely semantic garbage the whole time -- nothing in Stage 1
checked correctness, only speed, so this was never caught until now.
