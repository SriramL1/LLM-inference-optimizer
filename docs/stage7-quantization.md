# Stage 7: INT8 Weight Quantization

## Why this should help

Decode is memory-bandwidth-bound (Stage 1's central finding): every step
re-reads every weight matrix from GPU memory to produce one token, so
step time is dominated by *bytes of weights read*, not arithmetic. INT8
weights are half the bytes of fp16, so the dominant cost should roughly
halve. That's the whole thesis of this stage, and it's testable.

## Scheme: weight-only, per-output-channel, symmetric (W8A16)

- **Weight-only:** weights are stored as int8; activations stay fp16.
  No activation quantization, no calibration data, and far less accuracy
  risk than quantizing both sides.
- **Per-output-channel scales:** each row of a weight matrix `W` (shape
  `(out_features, in_features)`) gets its own scale,
  `scale[n] = max|W[n, :]| / 127`, so one outlier row can't wreck the
  precision of every other row.
- **Symmetric:** values map to `[-127, 127]`, zero maps to zero exactly,
  no zero-point bookkeeping.

## Kernel design

Because `scale[n]` is constant along the reduction axis,

```
y[m, n] = sum_k x[m, k] * (Wq[n, k] * scale[n])
        = scale[n] * sum_k x[m, k] * Wq[n, k]
```

so the kernel accumulates `x @ Wq^T` in fp32 using the raw integers
(int8 -> fp16 conversion is exact) and applies the scale once at the
end, rather than dequantizing every weight tile. Mathematically
identical, strictly less work.

**No `@triton.autotune`, deliberately.** Autotuning benchmarks candidate
configs on first call, which cannot happen inside CUDA graph capture
(Stage 6: a graphed engine's first call *is* the capture). Block sizes
come from a small fixed heuristic on `M` instead. The consequence is
honest and important: **this kernel is untuned for any specific GPU**.

## Integration

`QuantizedLinearInt8` replaces `nn.Linear` via **module replacement**
(setattr on the parent), not forward monkey-patching, per the Stage 3
finding that instance-level `.forward` reassignment is fragile in this
environment. `PagedEngine`, `GraphedDecodeEngine`, and
`ContinuousBatchingEngine` all call the projections as plain callables
looked up fresh each call, so they compose with a quantized model with
zero changes -- as does HuggingFace's own forward.

## Scope, stated plainly

- Quantized: all `nn.Linear` inside the transformer layers (q, k, v, o,
  gate, up, down) -- the bulk of the weights.
- **Not quantized: `embed_tokens` and `lm_head`.** For Qwen2.5-1.5B they
  are *tied* (one shared matrix), so quantizing `lm_head` would mean
  untying it, and logits precision matters more than the ~1-2ms it
  would save.
- **INT4 is deliberately deferred to a Stage 7b.** It needs nibble
  packing, group-wise scales, and a different kernel layout -- doing it
  alongside INT8 would double the debugging surface, the same reasoning
  behind the earlier 2a/2b and 4/4b splits.

## Correctness strategy

Two questions, tested separately on purpose (`tests/test_w8a16_kernel.py`):

1. **Is the kernel correct?** Checked against a reference that
   dequantizes the *same* int8 weights to fp32 and does an ordinary
   matmul. Both sides see identical quantized values, so any
   disagreement is a kernel bug, independent of how lossy quantization
   is. Shapes include Qwen2.5-1.5B's real projection sizes plus awkward
   ones (`K`, `N`, `M` not multiples of any block size) to exercise every
   masking path, across all three block-size tiers, with and without bias.
2. **How much accuracy does quantization itself cost?** Checked against
   the original fp16 `nn.Linear`. Lossy by design -- the test bounds it
   (~1% relative error expected for per-channel int8). A separate
   hard-guarantee test checks every dequantized weight is within half a
   quantization step of the original.

End to end (`tests/test_e2e_quantization.py`), applying the lesson from
Stages 2b and 3: **no exact-token-equality demand against fp16** over a
long generation -- any two numerically different implementations can flip
a near-tied argmax, and the flip compounds. Instead: logit cosine
similarity, top-1 agreement *rate* across a real prompt's positions, and
a check that memory actually dropped. The one exact-equality test is the
composition check, where it *is* the right bar: `PagedEngine` on the
quantized model vs. the same quantized model's own `generate()` -- same
weights on both sides.

```bash
pytest tests/test_w8a16_kernel.py tests/test_e2e_quantization.py -v
```

## Benchmarks: run in this order

```bash
python benchmarks/run_stage7_kernel.py   # first: is the kernel itself competitive?
python benchmarks/run_stage7_e2e.py      # then: does it help end to end?
```

**Why the kernel benchmark comes first:** an untuned Triton kernel is
competing against cuBLAS. A disappointing end-to-end speedup could mean
the *idea* (halve the bytes) doesn't pay off here, or that *this
kernel* is simply slower than cuBLAS -- the end-to-end numbers can't
distinguish those. The isolated benchmark reports effective weight
bandwidth (GB/s), which can: at decode shapes (`M=1`, memory-bound), if
the int8 kernel sustains similar GB/s to cuBLAS fp16 while reading half
the bytes, the time should drop ~2x.

**Reading the end-to-end benchmark:** it measures two decode paths.
The HuggingFace path at batch 1 is dominated by Python/launch overhead
(Stage 6's whole point), so expect a muted gain there. The
**graphed path** (Stage 6 composed with Stage 7) strips launch overhead
out and exposes the memory traffic -- the cleanest test of the stage's
thesis, and the number to weight most.

**What's guaranteed vs. what isn't:** the memory saving (roughly half of
the transformer layers' weights) is guaranteed by construction. The
*speedup* is not -- it depends on this kernel being competitive.

## Precautions recorded from the benchmark design

- **Triton JIT compilation happens before CUDA graph capture** (a
  throwaway eager decode step). Whether first-time compilation inside a
  capture region is safe was never actually verified in Stage 6 (its
  tests happened to run ungraphed code first, warming the cache).
  Avoiding the question is cheaper than answering it.
- **`max_context_len=512` for the graphed engine, not its 4096
  default.** The paged attention kernel loops `MAX_NUM_BLOCKS` times with
  masking (documented Stage 4 limitation), so a 4096 ceiling wastes
  iterations on every step and would dilute the effect being measured.
  It hits fp16 and int8 equally, but dilution still hides signal.

## Known limitations (intentional, for now)

- Kernel untuned; no autotune (see above). Costs prefill latency (TTFT 22 -> 44 ms at batch 1) and loses on small projection matrices.
- `lm_head` / embeddings stay fp16 (tied weights).
- INT8 only; INT4 is Stage 7b.
- Per-channel scale only -- no group-wise scales, which matter more at
  lower bit widths.

## Results (RTX 4060, Qwen2.5-1.5B-Instruct)

**Correctness:** all 55 tests passed (kernel vs. dequantized reference,
quantization-error bounds, end-to-end logit agreement, `PagedEngine`
composition on the quantized model).

**Memory:** transformer-layer weights 2499 MB -> 1252 MB (**2.00x
smaller**, 196 layers quantized); peak memory 2995 -> 1748 MB at batch 1.

### Isolated kernel vs. cuBLAS fp16 (`run_stage7_kernel.py`)

| layer | K x N | M=1 | M=8 | M=128 | M=1024 |
| --- | --- | --- | --- | --- | --- |
| q/o proj | 1536 x 1536 | 0.81x | 0.60x | 1.21x | 0.83x |
| k/v proj | 1536 x 256 | 0.62x | 0.64x | 0.71x | 0.59x |
| gate/up proj | 1536 x 8960 | **2.19x** | **2.11x** | 1.11x | 0.92x |
| down proj | 8960 x 1536 | **2.39x** | **1.72x** | 0.90x | 0.91x |

The thesis holds exactly where it should: on the **large** matrices at
decode-shaped `M`, the int8 kernel sustains ~235-255 GB/s of effective
bandwidth while reading half the bytes, giving 1.7x-2.4x. It loses on
the **small** matrices (k/v proj, q/o proj) and on large-`M`
(compute-bound, prefill-shaped) rows.

Likely explanation for the small-matrix losses (a hypothesis, not
tested): at 1536x256 or 1536x1536, even fp16 achieves only 24-98 GB/s,
so the call is dominated by fixed launch/latency cost rather than
bandwidth, and the untuned fixed block sizes launch very few programs
for small `N`. Fixing that is a tuning job (more programs / split-K for
skinny shapes), which this stage deliberately did not do.

### End to end (`run_stage7_e2e.py`)

| path | fp16 | int8 | ratio |
| --- | --- | --- | --- |
| HF path, batch 1 (decode tok/s) | 52.6 +/- 5.5 | 31.9 +/- 14.9 | 0.61x |
| HF path, batch 8 (decode tok/s) | 302.3 +/- 111.3 | 365.6 +/- 32.2 | 1.21x |
| **Graphed path, batch 1 (decode tok/s)** | **68.4 +/- 0.5** | **104.3 +/- 0.2** | **1.52x** |
| HF path, batch 1 TTFT | 22.1 ms | 43.8 ms | 0.50x |

How to read these honestly:

- **The graphed number is the one to trust.** Its trial-to-trial spread
  is tiny (+/-0.5, +/-0.2), and it is the path where launch overhead is
  stripped out so memory traffic dominates -- the cleanest test of the
  stage's thesis. **1.52x**, not the ideal 2x, because only the
  transformer-layer linears are quantized (embeddings/lm_head, attention,
  norms and the KV-cache reads are unchanged) and the small projections
  gain nothing, per the table above.
- **The HF-path batch-1 slowdown (0.61x) is real in direction but noisy
  in size** (+/-14.9 on a 31.9 mean). Likely cause (untested): that path
  is CPU/launch-bound, and a Triton kernel launch costs more Python-side
  than a cuBLAS call, across 196 layers. CUDA graphs remove exactly that
  cost, which is consistent with the graphed path winning. TTFT doubling
  (22 -> 44 ms) is consistent with the untuned kernel losing at
  prefill-sized `M`.
- **The batch-8 "1.21x" should not be quoted as a win:** the fp16
  baseline's own spread is +/-111 tok/s, so the difference is within noise.
- The graphed fp16 baseline here (68.4 tok/s) is not comparable to
  Stage 6's 51.4 tok/s: this benchmark uses `max_context_len=512`
  instead of the 4096 default.

### Takeaway

Halving weight bytes pays off for decode when launch overhead is out of
the way (graphed, 1.52x) and on the large MLP matrices (up to 2.4x
isolated), costs prefill latency, and a naive untuned kernel is a net
loss on the eager path at batch 1. Real systems get both by tuning per
shape and using INT8 only where it wins.
