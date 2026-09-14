"""
Stage 4b end-to-end correctness check.

PagedEngine's greedy generation must match a real model.generate() call
(reference: "sdpa", per the Stage 3 finding about "eager" producing NaN
in this environment) token-for-token, on a real prompt. This is the
test that actually matters here -- test_paged_attention_kernel.py
already validated the kernel and manager against synthetic data;
this validates the full manual forward-pass reimplementation (RoPE,
projections, layer wiring) against the real model, end to end.

Two prompt lengths: one shorter than a single page (block_size=16) and
one spanning multiple pages during prefill, so page-boundary handling
gets exercised during both prefill and the decode steps that follow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.engine.model_loader import load_model
from src.engine.paged_engine import PagedEngine
from src.kv_cache.block_manager import PagedKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE = 16

SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "The history of computing spans many decades, from early mechanical "
    "calculators to modern electronic computers. Charles Babbage designed "
    "the Analytical Engine in the 19th century, though it was never fully "
    "built in his lifetime."
)


def _build_paged_engine(loaded):
    config = loaded.model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    manager = PagedKVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=BLOCK_SIZE,
        max_pages=256,
        device="cuda",
        dtype=torch.float16,
    )
    return PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")


@pytest.mark.parametrize("prompt", [SHORT_PROMPT, LONG_PROMPT])
def test_paged_engine_matches_real_generation(prompt):
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    max_new_tokens = 16

    with torch.inference_mode():
        ref_out = loaded.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            repetition_penalty=1.0,  # neutralize -- this model's generation_config defaults to
                                     # 1.1, which generate() applies even with do_sample=False.
                                     # PagedEngine implements plain greedy argmax with no
                                     # repetition penalty, so the reference must match that
                                     # to be a fair comparison.
        )
    ref_new = ref_out[0, prompt_len:].tolist()

    engine = _build_paged_engine(loaded)
    paged_out = engine.generate(input_ids, max_new_tokens=max_new_tokens, seq_id=0)
    paged_new = paged_out[0, prompt_len:].tolist()

    assert paged_new == ref_new, (
        f"PagedEngine diverged from real generation.\n"
        f"ref:   {loaded.tokenizer.decode(ref_new)}\n"
        f"paged: {loaded.tokenizer.decode(paged_new)}"
    )
