"""
Stage 7 end-to-end checks.

Quantization is lossy by design, so these deliberately do NOT demand
exact token-for-token equality with the fp16 model over a long
generation (the lesson from Stage 2b / Stage 3: any two numerically
different implementations can flip a near-tied argmax, and the flip
compounds over autoregressive steps). Instead:

1. Logit similarity (cosine) and top-1 agreement RATE across every
   position of a real prompt -- bounds the accuracy cost.
2. Memory actually dropped -- the guaranteed benefit of this stage.
3. Composition: PagedEngine running on the quantized model must agree
   with the quantized model's own HuggingFace generate() -- same
   (quantized) weights on both sides, so exact agreement IS the right
   expectation here, unlike comparing against fp16.
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
from src.quantization.int8 import quantize_model_int8, QuantizedLinearInt8

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
    """If this fails, nothing below is testing quantization at all."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    stats = quantize_model_int8(loaded.model)

    num_layers = loaded.model.config.num_hidden_layers
    assert stats["layers_quantized"] == num_layers * 7, (
        f"expected 7 linears per layer (q,k,v,o,gate,up,down) x {num_layers} layers, "
        f"got {stats['layers_quantized']}"
    )
    assert isinstance(loaded.model.model.layers[0].self_attn.q_proj, QuantizedLinearInt8)
    assert stats["bytes_after"] < 0.6 * stats["bytes_before"]


def test_quantized_logits_close_to_fp16():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(LONG_PROMPT, return_tensors="pt").input_ids.to("cuda")

    with torch.inference_mode():
        fp16_logits = loaded.model(input_ids=input_ids).logits.float()

    quantize_model_int8(loaded.model)

    with torch.inference_mode():
        int8_logits = loaded.model(input_ids=input_ids).logits.float()

    assert not torch.isnan(int8_logits).any(), "quantized model produced NaN"

    cosine = torch.nn.functional.cosine_similarity(fp16_logits, int8_logits, dim=-1)
    assert cosine.mean().item() > 0.99, f"mean logit cosine similarity {cosine.mean().item():.4f}"

    agreement = (fp16_logits.argmax(-1) == int8_logits.argmax(-1)).float().mean().item()
    assert agreement >= 0.8, f"top-1 agreement with fp16 only {agreement:.1%}"


def test_paged_engine_on_quantized_model_matches_quantized_generate():
    """Composition check. Both sides use the SAME quantized weights, so
    unlike the fp16-vs-int8 comparison above, exact agreement is the
    right bar. repetition_penalty=1.0 for the same reason as Stage 4b:
    generate() otherwise applies this model's default 1.1 even in greedy
    mode, which PagedEngine (plain greedy) doesn't implement."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    quantize_model_int8(loaded.model)

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
        f"PagedEngine on quantized model diverged from the same model's generate().\n"
        f"generate(): {loaded.tokenizer.decode(ref_new)}\n"
        f"paged:      {loaded.tokenizer.decode(paged_new)}"
    )
