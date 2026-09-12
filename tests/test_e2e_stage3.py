"""
Stage 3 end-to-end correctness check.

Loads the model once, captures logits/generation on a real prompt, then
patches it in-place with the fused RMSNorm + SwiGLU kernels and checks
the same prompt produces matching output. Following the lesson learned
in Stage 2b: compare against a real, coherent prompt (not random tokens)
and check both a direct logits comparison (the precise signal) and
generation agreement (the practical one), rather than relying on a
single hard equality check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.engine.model_loader import load_model
from src.engine.fused_norm_activation_patch import patch_model_with_fused_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "The capital of France is"


def test_patch_actually_patches_something():
    """If this fails, the class-name matching in
    fused_norm_activation_patch.py doesn't recognize this model's
    module classes -- every other test in this file would then be
    trivially passing for the wrong reason (nothing changed)."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    counts = patch_model_with_fused_kernels(loaded.model)
    assert counts["rmsnorm"] > 0, "no RMSNorm modules were patched -- check class name matching"
    assert counts["mlp"] > 0, "no MLP modules were patched -- check class name matching"


def test_patched_logits_match_unpatched():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        unpatched_logits = loaded.model(input_ids=input_ids).logits

    counts = patch_model_with_fused_kernels(loaded.model)
    assert counts["rmsnorm"] > 0 and counts["mlp"] > 0

    with torch.inference_mode():
        patched_logits = loaded.model(input_ids=input_ids).logits

    torch.testing.assert_close(patched_logits, unpatched_logits, atol=2e-1, rtol=2e-1)

    sdpa_argmax = unpatched_logits.argmax(dim=-1)
    triton_argmax = patched_logits.argmax(dim=-1)
    mismatch_rate = (sdpa_argmax != triton_argmax).float().mean().item()
    assert mismatch_rate < 0.02, f"{mismatch_rate:.1%} of positions picked a different top token"


def test_patched_generation_matches_unpatched():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        unpatched_out = loaded.model.generate(input_ids, max_new_tokens=16, do_sample=False, num_beams=1)

    patch_model_with_fused_kernels(loaded.model)

    with torch.inference_mode():
        patched_out = loaded.model.generate(input_ids, max_new_tokens=16, do_sample=False, num_beams=1)

    unpatched_new = unpatched_out[0, prompt_len:].tolist()
    patched_new = patched_out[0, prompt_len:].tolist()

    assert patched_new == unpatched_new, (
        f"Patched generation diverged.\n"
        f"unpatched: {loaded.tokenizer.decode(unpatched_new)}\n"
        f"patched:   {loaded.tokenizer.decode(patched_new)}"
    )
