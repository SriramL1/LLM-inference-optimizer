# Stage 6: CUDA Graphs for Decode

## The problem this targets

Every previous stage made individual kernels faster or reduced memory
traffic. None of them address a different cost that specifically hurts
decode: each step launches dozens of tiny CUDA kernels (per layer: Q/K/V
projections, RoPE, attention, output projection, two norms, MLP's
matmuls) to process exactly **one new token**. At that scale, Python
dispatch and CUDA kernel *launch* overhead -- not the kernels' actual
GPU execution time -- can dominate wall-clock time. Making the kernels
themselves faster (Stages 2-4) doesn't touch this at all.

## The fix

CUDA graphs capture an entire sequence of kernel launches once, then
replay the whole thing with a single API call. Everything after the
first capture skips per-step Python overhead and per-kernel launch
overhead entirely.

## The hard constraint, and how Stage 4's design accidentally solved it

A captured graph's kernel launches reference **fixed memory addresses
and fixed tensor shapes** -- nothing about that recorded sequence can
change between replays. `PagedEngine.decode_step()` as written violates
this immediately: it builds a fresh, varying-size block-table tensor via
`torch.tensor(...)` every call, allocating new memory each time.
Capturing that literally would freeze the graph to whatever one
sequence's one specific context length happened to be at capture time.

The fix: pre-allocate static buffers once (`GraphedDecodeEngine.__init__`),
sized for a configurable `max_context_len`. Every decode call, for any
sequence, at any real context length, **overwrites those same buffers'
contents** (`.copy_()`/`.fill_()`, themselves ordinary graph-capturable
ops) rather than constructing new tensors, then replays the same
captured graph.

This works cleanly because `paged_attention_decode` already masks by the
*actual* `context_len` at runtime -- a Stage 4 design choice originally
motivated by supporting variable-length batches. It turns out to be
exactly what graph capture needs too: a graph captured with a fixed
`MAX_NUM_BLOCKS` safely handles any real context length up to that
maximum, paying only for some masked-out (wasted) iterations at shorter
lengths -- the same tradeoff Stage 4 already documented and accepted for
a different reason.

## What stays outside the graph, and why

Page **allocation** (`PagedKVCacheManager.reserve()`) has real control
flow -- whether this token starts a new page or not -- that cannot be
captured. It runs in plain Python before each graph replay. Per the
Stage 4b lesson, this is cheap CPU-only bookkeeping (the expensive case
there was GPU writes scaling with sequence length, which vectorization
already fixed), so leaving it outside the graph costs little.

## Scope: single-sequence decode, not combined with Stage 5

This deliberately doesn't combine with Stage 5's continuous batching.
A dynamically changing batch composition (sequence count varying every
iteration, as requests join and finish) is a substantially harder
graph-capture problem -- real systems handle it by capturing several
graphs for a few fixed batch-size "buckets" and dispatching to whichever
fits the current active batch size. Out of scope for this pass; a real
production engine would need this to get Stage 5 and Stage 6's benefits
simultaneously.

## Correctness strategy

`tests/test_graphed_decode.py` checks two things:
1. `GraphedDecodeEngine`'s output matches `PagedEngine`'s (ungraphed)
   output exactly, on a real prompt.
2. **A second, different sequence correctly reuses the same captured
   graph** -- the whole point of the static-buffer design is that one
   capture serves every subsequent call, not just repeated calls for the
   one sequence that happened to trigger capture.

```bash
pytest tests/test_graphed_decode.py -v
```

## A real bug found via careful elimination

The first working version failed decode's very first token, consistently,
regardless of variant tried. Three hypotheses were tested in order,
each ruled out with a targeted diagnostic rather than guessed away:

1. **The static-buffer refactor logic itself** -- ruled out by calling
   `_fill_static_buffers` + `_forward_body` directly in plain eager mode
   (no graph involved at all) and diffing against `PagedEngine`. Matched
   perfectly, proving the refactored logic was correct on its own.
2. **The side-stream warmup pattern** (copied from PyTorch's own CUDA
   graph docs, meant to let kernel autotuning settle before capture) --
   ruled out by removing it entirely. Same wrong result persisted.
3. **Triton/CUDA-graph incompatibility** (a real, commonly-cited risk
   for custom kernels) -- ruled out by swapping the Triton attention
   kernel for an equivalent, fully-vectorized pure-PyTorch computation
   inside the captured region. Produced the *exact same* wrong token,
   proving the attention implementation wasn't the cause either.

The actual cause, once those three were eliminated: **CUDA graph
capture's own execution pass does not guarantee a valid computed
result.** Capture primarily records the kernel-launch structure; a
correct result is only guaranteed after an actual `replay()`. The
original code read the capture call's own output directly, trusting it
as valid. Fixed by replaying once immediately after capture, before
returning that first logical decode step's result too -- a small,
surgical fix once the real cause was isolated, but one that would have
been very easy to misattribute to Triton or the warmup pattern without
methodically ruling each out first.

## Result

Real numbers on an RTX 4060, Qwen2.5-1.5B-Instruct, seq_len=128,
gen_len=64: **1.22x steady-state decode speedup** (42.1 -> 51.4 tok/s),
with a one-time warmup+capture cost of 47.4ms -- small relative to a
generation of any real length, and paid once per engine lifetime, not
per token.


The first draft of `benchmarks/run_stage6_e2e.py` tried to reuse the
captured graph across "fresh" sequences by swapping in a brand-new
`PagedKVCacheManager` each trial (to get a clean page pool). This is
wrong: a captured graph is bound to the *specific memory addresses* of
the K/V cache tensors that existed at capture time. Reassigning
`engine.manager` to a new manager afterward doesn't change what the
already-captured graph replays -- it would silently keep writing into
the *original* manager's tensors, not the new one's. Fixed by allocating
fresh sequences (new `seq_id`) within the **same** manager for
additional benchmark trials instead of swapping managers. Caught before
it produced a confusing or silently wrong benchmark number, but worth
recording since it's a natural mistake to make with this API and an
easy one to get wrong quietly.

## Benchmark

```bash
python benchmarks/run_stage6_e2e.py
```

Times decode steps only -- the one-time capture cost (which includes the
replay-after-capture fix above) is reported separately, since it's real
setup latency paid once per engine lifetime, not per generated token,
and would be misleading folded into a per-token throughput number.

## Known limitations (intentional, for now)

- Not combined with Stage 5 (see above).
- `max_context_len` is fixed at construction; a sequence exceeding it
  would need a larger `max_blocks` than the captured graph supports (not
  handled -- would need re-capture with a larger buffer).
- No mechanism for detecting when kernel autotuning or dtype changes
  would invalidate an already-captured graph.
