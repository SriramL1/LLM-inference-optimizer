# Stage 4b: PagedEngine — Wiring the Paged Cache Into Real Generation

## Scope and chosen approach

Two paths were scoped for this: implement a real `transformers.Cache`
subclass (works with `generate()` unmodified, but requires matching a
real, possibly version-specific interface contract), or extend a manual
per-layer loop we fully control (lower framework-surprise risk, at the
cost of not getting `generate()`/beam-search compatibility for free).
Chosen: the manual loop, given this project's track record of real
transformers-version surprises (Stage 2's `AttentionInterface` naming
collision, Stage 3's eager-attention bug) — proving correctness with
less exposure to an unfamiliar interface first.

## The correctness-risk strategy

Hand-reimplementing a transformer's forward pass is exactly the kind of
task where a subtle bug (wrong RoPE convention, wrong position id,
wrong tensor layout) produces plausible-looking but wrong output rather
than a crash — the hardest kind of bug to catch. `PagedEngine`
(`src/engine/paged_engine.py`) minimizes this by reusing HF's own tested
pieces wherever a from-scratch reimplementation would carry real risk:

- **q_proj/k_proj/v_proj/o_proj**: the actual `nn.Linear` layers, used
  directly.
- **Rotary embeddings**: the model's own `rotary_emb` module computes
  cos/sin; `apply_rotary_pos_emb` is located by **introspecting the
  actual installed `self_attn` module's own `__module__`**, not a
  hardcoded import path — the same "read the real source, don't assume
  the API" lesson Stage 2's naming collision taught, applied
  preemptively this time instead of after a debugging spiral.
- **input_layernorm / post_attention_layernorm / mlp**: called
  generically as plain callables. This means `PagedEngine` composes
  automatically with Stage 3 — if `patch_model_with_fused_kernels()` was
  already applied to the model, `PagedEngine` is calling
  `FusedRMSNorm`/`FusedMLP` without knowing or caring; if not, it's
  calling the originals. Either way, wiring is identical.

**Only the attention computation and KV-cache storage are genuinely
replaced** — that's the actual point of this stage. Prefill reuses
Stage 2's `flash_attention` kernel (ordinary self-attention over the
whole prompt); decode uses Stage 4's `paged_attention_decode` kernel
against the paged cache, allocating and writing new pages as the
sequence grows.

## Correctness strategy

`tests/test_e2e_paged_engine.py` compares `PagedEngine.generate()`
against a real `model.generate()` call (reference: `"sdpa"`, per the
Stage 3 finding about `"eager"`) token-for-token, on real prompts both
shorter and longer than a single page (`block_size=16`), so page-
boundary handling gets exercised during both prefill and the decode
steps that follow.

This is the test that actually matters here.
`test_paged_attention_kernel.py` already validated the kernel and
manager against synthetic data; this validates the much larger surface
area of the manual forward-pass reimplementation itself (RoPE
application, projection wiring, layer composition) against the real
model, end to end.

```bash
pytest tests/test_e2e_paged_engine.py -v
```

Given the size of this reimplementation relative to every previous
stage, **do not be surprised if this needs a debugging pass** — this
project's track record so far (Stage 2's three integration bugs, Stage
3's eager-attention discovery) suggests treating a clean first run as
the pleasant exception, not the expectation.

## A real finding along the way: `generate()` applies repetition_penalty even in greedy mode

The first real test run diverged from `model.generate()` after several
correct tokens (5/16 matching for a short prompt, 2/16 for a longer
one) -- a pattern worth reading carefully, since it's the signature of
*something* rather than nothing being wrong, just not necessarily in
`PagedEngine`. A structural bug (wrong RoPE, wrong masking) would
typically corrupt output from the very first generated token, not
several steps in.

Direct comparison confirmed it: `PagedEngine` and a manual step-by-step
loop using the real model's own `DynamicCache` agreed **perfectly**, 16
tokens in a row. `model.generate()` diverged from *both*. The cause:
Qwen2.5-1.5B-Instruct's `generation_config` sets `repetition_penalty:
1.1` -- and `generate()` applies this even with `do_sample=False`,
since it's a deterministic logits adjustment, not a sampling setting.
Neither `PagedEngine` nor a plain manual greedy loop implement
repetition penalty, so comparing either against an unmodified
`generate()` call is an apples-to-oranges comparison once the penalty
actually changes an argmax choice.

Fix: pass `repetition_penalty=1.0` to the reference `generate()` call to
neutralize it, making both sides genuinely pure greedy. This is a fix to
the *test's fairness*, not a lowering of the correctness bar --
`PagedEngine` was correct the whole time.

This also retroactively explains why Stage 1's equivalent test
(`test_baseline_engine.py`) never caught this: it uses
`attn_implementation="eager"`, which Stage 3 separately found produces
NaN logits in this environment. Both sides of that comparison were
equally broken, and NaN swallows any repetition-penalty adjustment
applied on top of it -- so the mismatch was masked by a different bug
entirely, not actually absent. Fixed there too, rather than relying on
that coincidence continuing to hold.

## A real performance bug found via benchmarking, not just testing

The first benchmark run showed `PagedEngine`'s TTFT at 4-5x the baseline
(95.6ms vs 23.0ms at seq_len=128; 332.1ms vs 62.2ms at seq_len=512) --
and critically, that gap *grew* with sequence length far faster than
Stage 2's own kernel benchmark would predict (the attention kernel
itself only gets ~0.04ms slower per call across that same size
difference). Something else was scaling with sequence length.

The cause: `PagedKVCacheManager.write()` wrote each token's K/V with a
Python `for` loop, one token at a time. For a 512-token prefill across
28 layers, that's 512 x 28 = 14,336 individual tiny GPU writes -- a cost
that scales with sequence length and dominated TTFT for anything but
very short prompts. Decode wasn't affected (always exactly 1 token per
write call), which is exactly why decode's numbers looked much closer
to baseline than TTFT's did from the start.

Fix: replace the per-token loop with a single vectorized (advanced-
indexing) tensor assignment covering all new tokens at once. Same
semantics, same correctness (all existing tests still pass unmodified),
dramatically less Python-dispatch/launch overhead. Measured impact:

| seq_len | TTFT before fix | TTFT after fix | baseline (sdpa) |
|---|---|---|---|
| 128 | 95.6ms | 31.1ms | 23.0ms |
| 512 | 332.1ms | 67.8ms | 62.2ms |

Post-fix, `PagedEngine` is within 9-35% of baseline TTFT -- the
remaining gap reflects ordinary "hand-rolled Python per-layer
orchestration vs. one compiled `nn.Module.forward()` call" overhead,
not a further bug. Decode remains ~30% slower than baseline
(unaddressed in this pass) for the same reason, compounded since it
repeats every generated token -- a reasonable target for future
optimization (e.g. caching the block-table tensor instead of rebuilding
it from a Python list every decode step), not something this stage's
scope required chasing further.

## Benchmark

```bash
python benchmarks/run_stage4b_e2e.py
```

**Honest framing, worth reading before interpreting the numbers**:
`PagedEngine`'s prefill reuses Stage 2's kernel, not SDPA/eager — so
this benchmark measures Stages 2 and 4 working together against the
Stage 1 baseline, not Stage 4 in isolation. Don't attribute the whole
TTFT/decode delta to paging alone; that's not what this comparison
shows. The benchmark also reports real memory usage via the manager's
own accounting (used/total pages, bytes) — that, not raw decode speed,
is this stage's actual point, consistent with the isolated memory
benchmark in Stage 4's own writeup.

## Known limitations (intentional, for now)

- Single sequence at a time (`batch=1` throughout `PagedEngine`) — true
  concurrent, variable-length batched serving (the scenario paging's
  memory savings actually matter most for) is Stage 5's job.
- No `generate()`/beam-search/sampling compatibility — greedy decoding
  only, via `PagedEngine`'s own `generate()` method, not HF's.
- No page eviction when the pool is exhausted (same limitation as
  Stage 4's manager itself).
