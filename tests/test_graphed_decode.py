"""
Stage 6 correctness check.

GraphedDecodeEngine's output must match PagedEngine's (ungraphed)
output exactly, on a real prompt -- graph capture should never change
what gets computed, only how fast it replays. Uses a real, coherent
prompt (per the Stage 2b lesson about random-token inputs and fp16
tie-breaking).

Also tests that a SECOND, different sequence (fresh seq_id, different
prompt) reuses the same captured graph correctly -- the whole point of
the static-buffer design is that one capture serves every subsequent
call, not just repeated calls for the same sequence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.engine.model_loader import load_model
from src.engine.paged_engine import PagedEngine
from src.engine.graphed_decode_engine import GraphedDecodeEngine
from src.kv_cache.block_manager import PagedKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 16

PROMPT_A = "The capital of France is"
PROMPT_B = "Water boils at a temperature of"


def _build_manager(config, max_pages=256):
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=BLOCK_SIZE, max_pages=max_pages, device="cuda", dtype=torch.float16,
    )


def test_graphed_decode_matches_ungraphed():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")
    input_ids = loaded.tokenizer(PROMPT_A, return_tensors="pt").input_ids.to("cuda")

    ungraphed_manager = _build_manager(loaded.model.config)
    ungraphed_engine = PagedEngine(loaded.model, loaded.tokenizer, ungraphed_manager, device="cuda")
    ungraphed_out = ungraphed_engine.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, seq_id=0)

    graphed_manager = _build_manager(loaded.model.config)
    graphed_engine = GraphedDecodeEngine(loaded.model, loaded.tokenizer, graphed_manager, device="cuda")
    graphed_out = graphed_engine.generate_graphed(input_ids, max_new_tokens=MAX_NEW_TOKENS, seq_id=0)

    prompt_len = input_ids.shape[1]
    ungraphed_new = ungraphed_out[0, prompt_len:].tolist()
    graphed_new = graphed_out[0, prompt_len:].tolist()

    assert graphed_new == ungraphed_new, (
        f"Graphed decode diverged from ungraphed.\n"
        f"ungraphed: {loaded.tokenizer.decode(ungraphed_new)}\n"
        f"graphed:   {loaded.tokenizer.decode(graphed_new)}"
    )


def test_captured_graph_reused_for_a_different_sequence():
    """The first sequence's decode call triggers capture; a second,
    different sequence must correctly reuse that SAME captured graph
    (not silently produce garbage from stale buffer contents)."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")

    manager = _build_manager(loaded.model.config)
    engine = GraphedDecodeEngine(loaded.model, loaded.tokenizer, manager, device="cuda")

    input_ids_a = loaded.tokenizer(PROMPT_A, return_tensors="pt").input_ids.to("cuda")
    input_ids_b = loaded.tokenizer(PROMPT_B, return_tensors="pt").input_ids.to("cuda")

    graphed_out_a = engine.generate_graphed(input_ids_a, max_new_tokens=MAX_NEW_TOKENS, seq_id=0)
    assert engine._graph is not None, "first generate_graphed call should have triggered capture"

    graphed_out_b = engine.generate_graphed(input_ids_b, max_new_tokens=MAX_NEW_TOKENS, seq_id=1)

    # Reference: same two prompts, run through plain PagedEngine.
    ref_manager_a = _build_manager(loaded.model.config)
    ref_engine_a = PagedEngine(loaded.model, loaded.tokenizer, ref_manager_a, device="cuda")
    ref_out_a = ref_engine_a.generate(input_ids_a, max_new_tokens=MAX_NEW_TOKENS, seq_id=0)

    ref_manager_b = _build_manager(loaded.model.config)
    ref_engine_b = PagedEngine(loaded.model, loaded.tokenizer, ref_manager_b, device="cuda")
    ref_out_b = ref_engine_b.generate(input_ids_b, max_new_tokens=MAX_NEW_TOKENS, seq_id=0)

    prompt_len_a = input_ids_a.shape[1]
    prompt_len_b = input_ids_b.shape[1]

    assert graphed_out_a[0, prompt_len_a:].tolist() == ref_out_a[0, prompt_len_a:].tolist()
    assert graphed_out_b[0, prompt_len_b:].tolist() == ref_out_b[0, prompt_len_b:].tolist()
