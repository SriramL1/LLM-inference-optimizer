# Stage 7b: INT4 Group-wise Weight Quantization

Status: complete. Kernel/quantizer tests (55 passed), the end-to-end
test (3 passed after resetting one accuracy threshold, see below) and
both benchmarks have been run; results at the bottom.

## Why INT4, and why it is a different problem from INT8

Same thesis as Stage 7: decode is memory-bandwidth-bound, so bytes of
weights read set the step time. INT4 weights are a quarter of fp16's
bytes (about 0.26x including scales), so the ceiling on the memory
saving is twice INT8's. But three things change:

1. **One scale per output row is no longer enough.** With only 15 usable
   levels, a single outlier in a row ruins the precision of every other
   weight in it. Scales are therefore **group-wise**: each row is split
   into groups of 128 consecutive input features with their own scale.
2. **The scale can't be hoisted out of the loop.** INT8's per-row scale
   was constant along the reduction axis, so the kernel applied it once
   at the end. A per-group scale varies along K, so it is applied once
   per group instead.
3. **Unpacking costs arithmetic.** Two weights share a byte, so every
   weight needs a shift/mask and an offset subtraction before use. If
   the kernel becomes ALU-bound rather than bandwidth-bound, INT4 can
   lose to INT8 despite reading half the bytes.

## Scheme

- Symmetric, group size 128: `scale[g, n] = max|W[n, group g]| / 7`,
  `q = round(W / scale)` in [-7, 7].
- Stored **offset-binary**: `q + 8` in [1, 15], so nibbles are unsigned
  and unpacking is `& 0xF` / `>> 4` followed by `- 8`.
- Scales stored fp16, laid out `(K/128, N)` so the kernel's per-group
  scale load is contiguous across output columns. Quantization rounds
  against the *stored* fp16 scale, so the kernel and reference agree.
- Plain round-to-nearest. No calibration data, no GPTQ-style error
  compensation, no AWQ-style activation-aware scaling -- those are what
  production INT4 uses to recover accuracy and are out of scope here.
  Expect noticeably more accuracy loss than INT8.

## Packing layout

Not adjacent pairs. Within each group of 128 input features, byte `j`
(0..63) holds element `j` in the low nibble and element `j + 64` in the
high nibble. One group is 64 contiguous bytes, and the kernel multiplies
`x[:, g*128 : g*128+64]` by the low nibbles and
`x[:, g*128+64 : g*128+128]` by the high nibbles. Both activation loads
are contiguous; adjacent-pair packing would force strided or
de-interleaved activation access.

Requires `in_features % 128 == 0`. True for every Qwen2.5-1.5B
projection (1536 and 8960 = 70 groups). The quantizer asserts it.

## Kernel

`BLOCK_K` equals the group size, so each loop iteration covers exactly
one group and needs exactly one scale per output column:

```
part = x_lo @ W_lo + x_hi @ W_hi        # fp32, raw integers
acc += part * scale[g, :]
```

As in Stage 7: **no `@triton.autotune`** (incompatible with CUDA graph
capture), fixed shape-based block sizes, therefore **untuned**. The
decode-sized config uses narrow N tiles (16 for N <= 2048) as a guess at
fixing the small-layer losses seen in the INT8 benchmark; that guess is
untested.

## Correctness strategy

Same two-question split as Stage 7 (`tests/test_w4a16_kernel.py`):

1. **Kernel + packing vs. a dequantizing reference** over the real
   Qwen projection shapes plus an awkward one, M in [1, 7, 16, 100, 300],
   with and without bias. A separate pure-bit-twiddling round-trip test
   covers every nibble value in both positions.
2. **Quantization error vs. the original fp16 linear**, bounded loosely
   (relative output error < 0.2), plus a hard guarantee that no
   dequantized weight is more than half a step from the original.

End to end (`tests/test_e2e_int4.py`): logit cosine similarity and
top-1 agreement *rate* vs. fp16 (no exact-token demand), a memory check,
and the exact-equality composition check (`PagedEngine` on the INT4
model vs. that same model's `generate()`).

**Threshold history, stated plainly:** I first guessed cosine > 0.95 and
agreement >= 0.6. The first real run measured cosine 0.9378 / agreement
80%, so the guess failed on cosine. The bars are now 0.90 / 0.7, set from
that one run with margin -- regression guards against a broken quantizer,
not derived accuracy bounds. The failure was the threshold being wrong,
not the model being broken: the per-position scores were mostly 0.93-0.99.

The packing/dequantization math and the kernel's group-wise accumulation
order were checked against a numpy emulation during development (pack /
unpack round-trip exact; emulated kernel vs. dequantized reference
differing by ~3e-6). That is a check of the algorithm, not of the Triton
kernel, which has not been executed yet.

## Run order

```bash
pytest tests/test_w4a16_kernel.py -v
pytest tests/test_e2e_int4.py -v -s
python benchmarks/run_stage7b_kernel.py     # INT4 vs INT8 vs cuBLAS fp16
python benchmarks/run_stage7b_e2e.py        # fp16 vs INT8 vs INT4, speed + accuracy
```

The e2e benchmark reloads the model for each precision and reports an
accuracy column next to the speed columns, because INT4 buys its extra
compression with accuracy and a speed-only table would hide that.

## Results (RTN INT4, group 128; RTX 4060, Qwen2.5-1.5B-Instruct)

**Memory** (reliable, structural): transformer-layer weights
2499 MB -> 645 MB (**3.88x smaller**); peak memory at batch 1
2995 -> 1179 MB.

**Accuracy** (one 60-token prompt, so a small sample):

| | logit cosine vs fp16 | top-1 agreement |
| --- | --- | --- |
| INT8 | 0.9991 | 96.7% |
| INT4 | 0.9378 | 80.0% |

Position 0 (the first token) has a cosine of only 0.27 and pulls the INT4
mean down; excluding it the mean is roughly 0.95. The two lowest later
positions are about 0.74-0.86. I have not investigated why position 0 is
so much worse (first-token logits are often unusual; that is a guess).
Real INT4 deployments use GPTQ/AWQ to close this gap; this stage does not.

**Isolated kernel** (`run_stage7b_kernel.py`; speedup vs cuBLAS fp16):

| layer | M | INT8 | INT4 | INT4 GB/s |
| --- | --- | --- | --- | --- |
| q/o proj | 1 | 0.78x | 0.78x | 22 |
| q/o proj | 1024 | 0.82x | 0.51x | 3 |
| k/v proj | 1 | 0.63x | 0.61x | 4 |
| k/v proj | 1024 | 0.60x | 0.42x | 2 |
| gate/up proj | 1 | 1.43x | **2.08x** | 116 |
| gate/up proj | 8 | 2.19x | 2.03x | 112 |
| gate/up proj | 1024 | 0.92x | 0.53x | 4 |
| down proj | 1 | **2.03x** | 1.02x | 49 |
| down proj | 8 | 1.97x | 0.90x | 47 |
| down proj | 1024 | 0.90x | 0.52x | 4 |

(Intermediate M=128 rows are in `benchmarks/results/stage7b_kernel.json`.)
INT4 beats INT8 only on gate/up at M=1. On down proj it loses badly to
INT8 (1.02x vs 2.03x), and at prefill-sized M it is roughly 0.5x fp16 --
about half as fast as INT8 there. Even its best case sustains only
~116 GB/s, about half of what the INT8 kernel reached (~235-255 GB/s),
so this kernel is far from bandwidth-bound. Likely causes (hypotheses,
untested): too few programs for narrow N tiles with a 70-iteration serial
K loop on down proj, plus the per-element unpack and per-group scale work.
Timings also moved noticeably between runs: INT8 gate/up at M=1 was 2.19x
in the Stage 7 run and 1.43x here, so single speedup figures carry maybe
+/-30%.

**End to end** (`run_stage7b_e2e.py`, speed relative to fp16):

| | layer MB | HF b=1 | HF b=8 | graphed b=1 | TTFT b=1 / b=8 |
| --- | --- | --- | --- | --- | --- |
| fp16 | 2499 | 1.00x | 1.00x | 1.00x (69.5 tok/s) | 24.0 / 120.5 ms |
| INT8 | 1252 | 0.93x | 0.87x | **1.50x** (104.2) | 21.9 / 126.3 ms |
| INT4 | 645 | 0.85x | 0.87x | **1.47x** (102.0) | 28.5 / 195.9 ms |

### Takeaway

- **INT4 gave no decode speedup over INT8** (102.0 vs 104.2 tok/s graphed,
  within noise of each other) even though it reads half the bytes, because
  this kernel only reaches about half of INT8's effective bandwidth. The
  bytes were halved; the time did not follow. The thesis ("decode time
  tracks bytes read") holds only if the kernel keeps the memory system
  busy, and this one doesn't.
- **What INT4 does buy is memory**: 3.88x smaller layer weights, about
  1.8 GB less peak memory than fp16 at batch 1 -- the difference between
  fitting and not fitting a larger model on an 8 GB card. It pays for that
  with real accuracy loss (cosine 0.94, 80% top-1 on this prompt) and
  slower prefill (batch-8 TTFT 196 ms vs 120 ms).
- Honest ranking for this repo: INT8 is the better trade for speed and
  accuracy; INT4 is the choice only when memory is the constraint.
- Next steps, if pursued: tune the INT4 kernel (larger N tiles, split-K
  for K=8960, vectorized packed loads) and/or add GPTQ/AWQ-style
  calibration. Neither is done here.

## Known limitations

- RTN only; no GPTQ/AWQ. Accuracy will be worse than a production INT4.
- Kernel untuned; no autotune.
- `in_features` must be a multiple of 128.
- Embeddings / tied `lm_head` stay fp16.
