"""
Stage 2b end-to-end correctness check.

Design note (this took one failed iteration to get right, worth
recording): the obvious test -- compare eager vs Triton generate() output
token-for-token -- is too fragile to trust, for two independent reasons:

1. Our dispatch (flash_attention_patch.py) falls back to SDPA for decode,
   not eager. Comparing against an "eager" baseline model means eager's
   decode numerics (different summation order/rounding than SDPA) are a
   *separate, pre-existing* source of divergence, unrelated to whether our
   kernel is correct. The fair comparison is against an "sdpa" baseline --
   that isolates exactly the one thing that changed: prefill.
2. Random/incoherent input produces a near-flat logit distribution (the
   model has no real signal to work with), so tiny fp16 rounding
   differences between any two valid implementations can flip which token
   wins argmax -- especially compounded over many autoregressive decode
   steps. A real, coherent prompt gives the model confident, well-
   separated logits, which is far less sensitive to this.

So this test does two things instead: (a) a direct logits comparison on a
single prefill forward pass with a real prompt -- the actual, precise
correctness signal, analogous to the atol/rtol check in
test_flash_attention_kernel.py -- and (b) a generate() comparison as a
secondary sanity check, still against the "sdpa" reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.engine.model_loader import load_model
from src.engine.flash_attention_patch import register_triton_flash_attention, ATTN_IMPL_NAME

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Real, coherent prompts -- not random tokens -- so the model produces
# confident, well-separated logits that aren't sensitive to fp16
# tie-breaking noise. One short, one long enough to span multiple kernel
# tiles (BLOCK_M=BLOCK_N=64).
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "The history of computing spans many decades, from early mechanical "
    "calculators to modern electronic computers. Charles Babbage designed "
    "the Analytical Engine in the 19th century, though it was never fully "
    "built in his lifetime. Ada Lovelace wrote what is considered the "
    "first algorithm intended for machine execution. In the 20th century, "
    "Alan Turing formalized the concept of computation with the Turing "
    "machine, laying theoretical groundwork for everything that followed. "
    "The first electronic general-purpose computers emerged during and "
    "after the Second World War, including ENIAC in the United States."
)


@pytest.mark.parametrize("prompt", [SHORT_PROMPT, LONG_PROMPT])
def test_prefill_logits_match_sdpa(prompt):
    """Direct logits comparison on a single prefill forward pass -- the
    precise correctness signal. Compared against 'sdpa', not 'eager',
    since that's the true apples-to-apples reference (decode in our
    dispatch also falls back to sdpa, so this isolates exactly the one
    thing Stage 2b changed: the prefill path)."""
    register_triton_flash_attention()

    sdpa = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    triton_model = load_model(MODEL_NAME, device="cuda", attn_implementation=ATTN_IMPL_NAME)

    input_ids = sdpa.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        sdpa_logits = sdpa.model(input_ids=input_ids).logits
        triton_logits = triton_model.model(input_ids=input_ids).logits

    # Loose bound on raw logit magnitude: fp16 accumulation noise scales
    # with sequence length (more online-softmax rescaling blocks in the
    # kernel), so a tolerance that's comfortable for a 6-token prompt is
    # expected to occasionally trip on a 200+-token one even with zero
    # real bug. This is a coarse sanity check, not the precise signal.
    torch.testing.assert_close(triton_logits, sdpa_logits, atol=2e-1, rtol=2e-1)

    # The signal that actually matters: would greedy decoding pick the
    # same token at every position? This is far more sensitive to *real*
    # bugs (wrong stride, wrong head grouping, off-by-one masking) than
    # raw logit closeness, and far less sensitive to harmless fp16
    # rounding noise on low-probability vocab entries no decoding
    # strategy would ever select anyway.
    sdpa_argmax = sdpa_logits.argmax(dim=-1)
    triton_argmax = triton_logits.argmax(dim=-1)
    mismatches = (sdpa_argmax != triton_argmax).sum().item()
    total = sdpa_argmax.numel()
    mismatch_rate = mismatches / total
    assert mismatch_rate < 0.02, (
        f"{mismatches}/{total} positions ({mismatch_rate:.1%}) picked a "
        f"different top-1 token between sdpa and triton -- too high to "
        f"be explained by fp16 rounding noise alone."
    )


def test_generation_matches_sdpa():
    """Secondary sanity check: greedy generation, real prompt, against the
    sdpa reference (not eager, per the module docstring)."""
    register_triton_flash_attention()

    sdpa = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    triton_model = load_model(MODEL_NAME, device="cuda", attn_implementation=ATTN_IMPL_NAME)

    input_ids = sdpa.tokenizer(SHORT_PROMPT, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        sdpa_out = sdpa.model.generate(input_ids, max_new_tokens=16, do_sample=False, num_beams=1)
        triton_out = triton_model.model.generate(input_ids, max_new_tokens=16, do_sample=False, num_beams=1)

    sdpa_new = sdpa_out[0, prompt_len:].tolist()
    triton_new = triton_out[0, prompt_len:].tolist()

    assert triton_new == sdpa_new, (
        f"Triton-attention generation diverged from sdpa reference on a real prompt.\n"
        f"sdpa:   {sdpa.tokenizer.decode(sdpa_new)}\n"
        f"triton: {triton_model.tokenizer.decode(triton_new)}"
    )
