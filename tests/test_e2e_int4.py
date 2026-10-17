"""
Stage 7b end-to-end checks (INT4).

Same structure and philosophy as tests/test_e2e_quantization.py: no exact
token equality against fp16 (lossy, and argmax flips compound), instead
logit cosine similarity + top-1 agreement RATE, a real memory check, and
an exact-equality composition check where both sides share the SAME
quantized weights.

The accuracy thresholds below are GUESSES, deliberately looser than INT8's
(cosine 0.99 / agreement 0.8) because plain round-to-nearest 4-bit with no
calibration (no GPTQ/AWQ) is meaningfully lossier. They have not been
calibrated against a measured run; the measured values are printed so the
thresholds can be set from data (run with -s).
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
from src.quantization.int4 import quantize_model_int4, QuantizedLinearInt4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE = 16

SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "The history of computing spans many decades, from early mechanical "
    "calculators to modern electronic computers. Charles Babbage designed "
    "the Analytical Engine in the 19th century, though it was never fully "
    "built in his lifetime. Ada Lovelace wrote what is considered the "
    "first algorithm intended for machine execution."
)


def test_quantizer_actually_replaces_layers():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    stats = quantize_model_int4(loaded.model)

    num_layers = loaded.model.config.num_hidden_layers
    assert stats["layers_quantized"] == num_layers * 7
    assert isinstance(loaded.model.model.layers[0].self_attn.q_proj, QuantizedLinearInt4)
    assert stats["bytes_after"] < 0.30 * stats["bytes_before"]


def test_quantized_logits_close_to_fp16():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(LONG_PROMPT, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        fp16_logits = loaded.model(input_ids=input_ids).logits.float()

    quantize_model_int4(loaded.model)

    with torch.inference_mode():
        int4_logits = loaded.model(input_ids=input_ids).logits.float()

    assert not torch.isnan(int4_logits).any(), "quantized model produced NaN"

    cosine = torch.nn.functional.cosine_similarity(fp16_logits, int4_logits, dim=-1)
    agreement = (fp16_logits.argmax(-1) == int4_logits.argmax(-1)).float().mean().item()
    print(f"\n[int4] mean logit cosine = {cosine.mean().item():.4f}, top-1 agreement = {agreement:.1%}")

    assert cosine.mean().item() > 0.90, f"mean logit cosine similarity {cosine.mean().item():.4f}"
    assert agreement >= 0.7, f"top-1 agreement with fp16 only {agreement:.1%}"


def test_paged_engine_on_quantized_model_matches_quantized_generate():
    """Composition check; same quantized weights on both sides, so exact
    agreement is the right bar. repetition_penalty=1.0 as in Stage 4b."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    quantize_model_int4(loaded.model)

    input_ids = loaded.tokenizer(SHORT_PROMPT, return_tensors="pt").input_ids.to("cuda")
    prompt_len = input_ids.shape[1]
    max_new_tokens = 8

    with torch.inference_mode():
        ref_out = loaded.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, repetition_penalty=1.0,
        )
    ref_new = ref_out[0, prompt_len:].tolist()

    config = loaded.model.config
    manager = PagedKVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        block_size=BLOCK_SIZE, max_pages=128, device="cuda", dtype=torch.float16,
    )
    engine = PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    paged_out = engine.generate(input_ids, max_new_tokens=max_new_tokens, seq_id=0)
    paged_new = paged_out[0, prompt_len:].tolist()

    assert paged_new == ref_new, (
        f"PagedEngine on INT4 model diverged from the same model's generate().\n"
        f"generate(): {loaded.tokenizer.decode(ref_new)}\n"
        f"paged:      {loaded.tokenizer.decode(paged_new)}"
    )
