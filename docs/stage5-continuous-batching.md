# Stage 5: Continuous Batching

## What this builds on

This stage needed almost no new kernel work, because Stage 4 already
built the piece that makes it possible: `paged_attention_decode`
already supports variable-length batches (validated in Stage 4's own
correctness tests), and `PagedKVCacheManager` already has
`batch_block_table_tensor()`/`batch_context_lens_tensor()` for exactly
this purpose. Stage 5 is almost entirely new *engine and scheduling*
logic around infrastructure that already existed.

## The problem with static batching

Without continuous batching, a batch of requests all start and finish
together -- if one request needs 500 tokens and the rest need 20, every
other request's GPU slot sits idle waiting for the long one to finish
(or the system processes requests one at a time, wasting the batching
throughput gains Stage 1 already measured: decode scales near-linearly
with batch size, ~47 -> 384 tok/s from batch 1 to 8).

## The fix: iteration-level scheduling

`ContinuousBatchingScheduler` (`src/engine/scheduler.py`) admits a new
request the moment a slot frees up (a previous request finished),
rather than waiting for the whole batch to complete. Each iteration:
1. Admit pending requests if there's room (each admitted request's
   prefill runs individually).
2. Run one **batched decode step** across every currently active
   sequence via `ContinuousBatchingEngine.decode_step_batched()` --
   sequences at completely different context lengths, batched into a
   single kernel call.
3. Any sequence that just hit its `max_new_tokens` is removed and its
   pages freed, opening a slot for the next pending request.

`ContinuousBatchingEngine` extends `PagedEngine` rather than duplicating
it -- prefill is unchanged; the only new method is
`decode_step_batched()`, built by generalizing `PagedEngine.decode_step()`
from one sequence to a batch (each with its own position id, block
table, and context length).

## Explicitly out of scope: chunked prefill

Real systems (vLLM) also mix prefill's many new tokens into the *same*
batched kernel call as other sequences' decode steps ("chunked
prefill"), which needs more sophisticated masking than this project
attempts. Here, prefill always runs alone, one sequence at a time; once
prefilled, a sequence joins the batched decode pool. This is a real
scope limitation, not an oversight -- iteration-level scheduling with
dynamic admission is the core continuous-batching mechanism and the
thing that actually drives the throughput number below; chunked prefill
is a further refinement on top of it.

## Correctness strategy

The critical invariant: a sequence's output must be **identical**
whether it runs alone or batched with others -- if batching changed
results, "free throughput" would actually mean "wrong answers
sometimes," which would make the whole optimization worthless.
`tests/test_continuous_batching.py` checks two things:
1. Multiple sequences submitted together, each compared against running
   standalone via `PagedEngine`.
2. **Staggered arrival** -- a request added mid-stream, after other
   sequences are already several decode steps in. This is the actual
   scenario continuous batching exists for, and the case a naive
   admission implementation could plausibly get wrong (e.g. if adding a
   new sequence disturbed already-active sequences' cache state).

```bash
pytest tests/test_continuous_batching.py -v
```

## Benchmark

```bash
python benchmarks/run_stage5_e2e.py
```

Compares total wall-clock time to process a fixed set of **varying-
length** requests (uniform-length requests would understate the real
advantage, since real traffic never arrives uniform) two ways:
sequentially (one full request at a time, even though it still uses
Stage 4's paged cache) versus via the scheduler. The aggregate
throughput and speedup numbers here are what actually demonstrates this
stage's value -- Stage 1 already proved batching helps decode in
isolation; this proves it holds up under realistic, staggered,
variable-length arrival, not just a pre-formed uniform batch.

## Known limitations (intentional, for now)

- No chunked prefill (see above).
- No EOS-based early stopping -- every request runs for its full fixed
  `max_new_tokens`.
- FCFS admission only, no priority or fairness policy.
- No preemption/eviction when the page pool is exhausted -- assumes
  `max_batch_size` and `max_pages` are sized generously enough to avoid
  that case, consistent with Stage 4's own stated scope.
