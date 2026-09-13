# Stage 4: Paged KV-Cache Manager

## Scope of this pass

This stage is split the same way Stage 2 was (kernel first, model
integration as a follow-up): this covers the **memory manager and paged
attention decode kernel, validated in isolation**. Wiring this into the
real model's generation loop (replacing HF's `DynamicCache` entirely) is
a larger integration task, deferred to a Stage 4b follow-up once this is
confirmed correct on your hardware -- same reasoning as Stage 2/2b.

## The problem

HuggingFace's default KV-cache (`DynamicCache`) is one contiguous,
growing tensor per sequence. Simple, but wasteful for real serving:
batching sequences of different lengths together either forces
over-allocation (reserve for a worst-case length) or expensive
reallocation as sequences grow. Neither scales to serving many
concurrent requests efficiently.

## The fix: paging (vLLM's PagedAttention)

Split the KV-cache into fixed-size **pages** (default: 16 tokens each,
matching vLLM's typical default), allocate them on demand as a sequence
grows, and keep a per-sequence **block table** mapping logical token
positions to physical page indices -- the same idea as OS virtual memory
paging. Memory is only used for tokens that actually exist, plus at most
`block_size - 1` tokens of waste in a sequence's partially-filled last
page.

## Design

- `BlockAllocator` (`src/kv_cache/block_manager.py`): free-list page
  allocator. Just hands out and reclaims integer page ids -- knows
  nothing about K/V values.
- `BlockTable`: per-sequence bookkeeping -- which physical pages hold its
  tokens, in order, and how many tokens are actually valid.
- `PagedKVCacheManager`: owns the allocator, all sequences' block
  tables, and the actual physical K/V storage (one tensor pair per
  layer, shape `(max_pages, block_size, num_kv_heads, head_dim)`).
  Block tables are **shared across layers** for a given sequence --
  every layer processes the same tokens in lockstep, so "logical block 3
  of sequence 7" means the same physical page index in every layer's
  storage. Only the K/V *values* differ per layer, not the allocation
  bookkeeping. `reserve()` is called once per forward step (allocating
  pages as needed); `write()` is called once per layer using that same
  reservation.

## The kernel

`src/kernels/paged_attention_triton.py` handles decode's actual shape:
one new query token per sequence, attending over that sequence's full
cached context, which now lives scattered across pages rather than one
flat buffer. The kernel reads the block table, gathers each page's K/V
via a runtime-computed (not compile-time-tiled) address, and runs the
same online-softmax accumulation Stage 2's flash attention kernel uses --
just over page-sized chunks read through a gather instead of a clean
tile.

**Scope: decode only.** Prefill still uses Stage 2's kernel (or SDPA) on
the token span being processed for the first time -- paging only matters
once tokens need to persist in a cache across steps.

## Correctness strategy

`tests/test_paged_attention_kernel.py` checks against a naive reference
that gathers pages into a contiguous tensor in plain PyTorch first, then
runs ordinary attention -- the same "obviously correct, obviously slow"
reference pattern used throughout this project. Test cases specifically
include:
- Exactly one full page, and one-page-plus-one-token (boundary case)
- A context shorter than a single page (partial-page masking)
- **A batch of sequences with different lengths in the same call** --
  this is the case that actually matters; a test suite that only used
  uniform-length batches would validate a much narrower (and less
  interesting) slice of this stage's actual purpose.

Manager-level tests separately check page allocation/freeing accounting
and that the allocator raises cleanly when the pool is exhausted, rather
than silently corrupting memory.

Run with:
```bash
pytest tests/test_paged_attention_kernel.py -v
```

## Benchmarks: two, deliberately answering different questions

**Speed** (`benchmarks/run_stage4_kernel.py`): paged decode kernel vs.
plain SDPA over an equivalent contiguous cache, same data, same context
length. **Expect the paged kernel to be roughly on par with or somewhat
slower than SDPA here** -- gathering scattered pages via a block table is
inherently more work than reading one flat tensor. This matches vLLM's
own reported tradeoff; it is not a failure of this implementation.

Real result on an RTX 4060: paged is 10-64% slower at batch=1 (as
expected, worsening with longer context), but *faster* than SDPA at
batch=8 and batch=32 (down to 0.81x at batch=32, ctx_len=2048). Take
that crossover with a grain of salt, though: the SDPA baseline here
includes a `repeat_interleave()` call inside the timed region to
materialize expanded K/V for GQA, while the paged kernel computes the KV
head index via integer division internally with no materialization
step. That repeat_interleave cost grows with batch size, so part of
paged's apparent win at scale may reflect an inefficiency specific to
this SDPA baseline's construction, not a general proof that gathered
access beats contiguous access at scale. The batch=1 numbers (no such
confound) are the more trustworthy read of the raw gather-overhead
question.

**Memory** (`benchmarks/run_stage4_memory.py`): the comparison that
actually demonstrates paging's point -- a batch of sequences with
realistic, wildly different lengths, comparing total memory a naive
system would need (pre-allocating every sequence for some worst-case
`max_seq_len`) against paging's actual usage (only what's used, rounded
up to the nearest page). This is pure allocation-policy arithmetic, not
a kernel timing test.

```bash
python benchmarks/run_stage4_kernel.py
python benchmarks/run_stage4_memory.py
```

## Known limitations (intentional, for now)

- No model integration yet (Stage 4b) -- this validates the manager and
  kernel in isolation with synthetic data.
- No page eviction/preemption policy (what happens when the pool is
  exhausted and a running sequence needs more space) -- currently just
  raises. Real systems (vLLM) implement swapping or recomputation here;
  out of scope for this pass.
- No prefix-sharing / copy-on-write between sequences (e.g. shared system
  prompts across requests) -- a real vLLM feature this project doesn't
  attempt yet.
- Single fixed block size, no per-workload tuning.
- `MAX_NUM_BLOCKS` (the kernel's loop trip count) is one compile-time
  value per launch, sized for the *longest* sequence in the batch. A
  short sequence sharing a batch with a much longer one still executes
  the same number of loop iterations -- masked out as invalid, but not
  skipped -- so a highly skewed batch (one 4000-token sequence next to
  several 50-token ones) wastes real cycles on the short sequences. Real
  vLLM addresses this with more sophisticated dispatch; out of scope
  here.
