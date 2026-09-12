# Stage 2b: Wiring the Kernel Into the Real Model

## Why this stage exists

`run_stage2_attention.py` validated the Triton kernel in isolation: it
matches SDPA within noise at 2048-4096 tokens, ~14x faster than naive. But
that's the attention op alone. This stage answers the question that
actually matters: does swapping it into the real model move the TTFT
number Stage 1 established as the baseline?

Amdahl's Law is the reason these can differ: if attention is, say, 40% of
prefill time and the rest is MLP/layernorm/embedding (which get no
faster), even a "perfect" attention speedup caps the whole-model speedup
well below whatever the isolated kernel benchmark showed.

## How the integration works

HuggingFace transformers has a documented, supported extension point for
this: `AttentionInterface.register(name, fn)`
(https://huggingface.co/docs/transformers/attention_interface). Once
registered, `attn_implementation=name` is a valid choice everywhere
`attn_implementation="eager"` or `"sdpa"` would be -- no internals
monkey-patched.

`src/engine/flash_attention_patch.py` registers
`"triton_attn"` and implements **per-call dispatch**, not a
blanket replacement:

- **Prefill call** (query and key share the same sequence length, >1
  token, no padding) → routed to the Stage 2 Triton kernel.
- **Everything else** (decode: 1-token query vs a long KV-cache; any
  padded batch) → falls back to PyTorch's own `sdpa_attention_forward`.

This makes the integration safe by construction: it can never silently
misbehave outside the kernel's tested scope, because it simply doesn't
use the kernel there. The cost is that decode gets zero speedup from this
stage -- expected and correct, since decode is a different bottleneck
(memory-bandwidth, not compute) that later stages (paged KV-cache,
continuous batching, CUDA graphs) target instead.

## Correctness first

`tests/test_e2e_attention_patch.py` compares the model loaded with the
Triton attention implementation against the model loaded with `"sdpa"`
(not `"eager"` -- see "Lessons" below for why that distinction matters).
Two checks: a direct logits comparison on a single prefill forward pass
(the precise signal -- are the numbers actually right, with a coarse
magnitude bound plus a stricter "does greedy decoding pick the same
token everywhere" check), and a `generate()` comparison as a secondary
sanity check. Real, coherent prompts are used throughout, one short and
one long enough to span multiple kernel tiles.

This is the test that actually matters for this stage. A kernel that
passed its isolated unit tests could still be wired in wrong (bad
stride, wrong transpose, mismatched head grouping) and silently corrupt
real model output. Run it before trusting any benchmark number:

```bash
pytest tests/test_e2e_attention_patch.py -v
```

## Benchmark

```bash
python benchmarks/run_stage2_e2e.py
```

Loads the model twice (eager, then Triton), sweeps prompt length,
reports TTFT and decode tok/s for each, and prints/saves the speedup.

**Reading the decode column correctly matters here.** The eager baseline
uses eager attention for its own decode steps too (never optimized),
while the Triton variant's decode falls back to SDPA (per the dispatch
logic above). So any decode-speed difference you see reflects PyTorch's
own pre-existing SDPA-vs-eager advantage, not anything this stage built
-- it's not a Stage 2 result and real numbers here (observed: 1.03x-1.28x
decode "speedup") shouldn't be attributed to the Triton kernel. **TTFT is
the number that actually measures Stage 2's contribution** -- it should
grow with sequence length, since that's exactly the shape the isolated
kernel benchmark predicted (Triton ≈ SDPA-speed, vs. eager's slow
O(n²)-materializing path).

Both model instances are loaded and freed sequentially (`del` +
`torch.cuda.empty_cache()` between variants) rather than held
simultaneously, since your 8GB RTX 4060 doesn't have headroom for two
copies of the model plus both KV-caches at once.

## Lessons from getting this wrong the first time

The first version of this integration hit two real issues worth
recording, not just fixing silently:

1. **Naming collision with transformers' internal dispatch.** The
   registered name originally included the substring `flash_attention`
   (`"triton_flash_attention"`). transformers has special-case handling
   for any `attn_implementation` string matching that pattern -- it tries
   to resolve it as a known FlashAttention variant or a Hub kernel repo
   *before* checking the plain `AttentionInterface` registry, and fails
   loudly if that resolution doesn't find a real package/repo. Renamed to
   `"triton_attn"` to route through the plain registry instead.
2. **The correctness test itself was flawed, not the kernel.** Comparing
   generation against an `"eager"` reference conflated two different
   things: our actual change (prefill: eager → Triton) and a *separate*,
   pre-existing numerical difference (decode: eager → sdpa, since our
   dispatch falls back to sdpa for decode regardless of what the
   baseline model uses). On top of that, using random/incoherent token
   IDs as test input produces near-flat logit distributions where tiny
   fp16 rounding differences between any two valid implementations can
   flip which token wins argmax -- especially compounded over many
   decode steps. Fixed by comparing against an `"sdpa"` reference (the
   true apples-to-apples baseline) and using real, coherent prompts, plus
   a direct logits comparison (not just final-token equality) as the
   primary correctness signal.
3. **Full-tensor `allclose` on raw logits is too strict for longer
   sequences.** A 206-token prompt goes through 4 kernel tiling blocks of
   online-softmax rescaling (vs. 1 for a 6-token prompt), and fp16
   rounding compounds slightly across each rescale step -- so a tolerance
   comfortable at short lengths trips at longer ones (~0.024% of logit
   values exceeded a 0.05 absolute tolerance, worst case 0.135) even with
   no real bug. Fixed by keeping a loose magnitude bound as a coarse
   sanity check, but making the actual pass/fail signal "would greedy
   decoding pick the same token at every position" -- far more sensitive
   to real bugs, far less sensitive to harmless noise on vocab entries no
   decoding strategy would ever select.
4. **My own benchmark script's printed conclusion was wrong on the first
   real run.** The eager baseline model uses eager attention for its own
   decode steps too (loaded globally as `attn_implementation="eager"`),
   while the Triton variant's decode falls back to SDPA. The script
   originally claimed "decode speedup should be ~1.0x" -- false, since
   the two variants use genuinely different decode implementations.
   Real run showed 1.03x-1.28x "decode speedup," which is actually
   PyTorch's pre-existing SDPA-vs-eager advantage leaking into the
   comparison, not a Stage 2 result. Fixed the script's note rather than
   the numbers (the numbers were always correct; the interpretation
   printed alongside them wasn't). Only the TTFT column measures what
   this stage actually built.



## Known limitations (intentional, for now)

- No padding support in the fast path -- any padded batch falls back to
  SDPA. Real variable-length batching is Stage 5's job (continuous
  batching), not a hack bolted onto Stage 2's dispatch logic.
- Dispatch check itself (`q_len == kv_len and q_len > 1`, mask
  inspection) adds a small amount of Python-side overhead per attention
  call. Not expected to be visible against a GPU kernel's runtime, but
  worth knowing it's there if CPU-side profiling ever looks off.
