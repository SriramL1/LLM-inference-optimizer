"""
Stage 5 correctness checks.

The critical invariant for continuous batching: a sequence's output must
be identical whether it runs alone or interleaved with other sequences
in the same batched decode steps. If batching changed any sequence's
result, the whole point of the optimization (free throughput, zero
semantic cost) would be false.

Two things tested:
1. Multiple sequences submitted together, run via the scheduler, each
   compared against running standalone via PagedEngine.
2. A sequence added mid-stream (staggered arrival, not all requests
   present at t=0) -- the actual scenario continuous batching exists
   for, and a case a naive implementation could plausibly get wrong
   (e.g. if admission logic disturbed already-active sequences' state).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.engine.model_loader import load_model
from src.engine.paged_engine import PagedEngine
from src.engine.continuous_batching_engine import ContinuousBatchingEngine
from src.engine.scheduler import ContinuousBatchingScheduler
from src.kv_cache.block_manager import PagedKVCacheManager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 12

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
]


def _build_manager(config, max_pages=512):
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return PagedKVCacheManager(
        num_layers=config.num_hidden_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=BLOCK_SIZE, max_pages=max_pages, device="cuda", dtype=torch.float16,
    )


def _run_standalone(loaded, prompt: str, max_new_tokens: int) -> list:
    """Runs one prompt alone through PagedEngine -- the ground truth
    each batched result is checked against."""
    manager = _build_manager(loaded.model.config)
    engine = PagedEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    input_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    out = engine.generate(input_ids, max_new_tokens=max_new_tokens, seq_id=0)
    return out[0, input_ids.shape[1]:].tolist()


def test_batched_matches_standalone_for_all_sequences():
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")

    standalone_results = [_run_standalone(loaded, p, MAX_NEW_TOKENS) for p in PROMPTS]

    manager = _build_manager(loaded.model.config)
    engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    scheduler = ContinuousBatchingScheduler(engine, max_batch_size=len(PROMPTS))

    prompt_lens = []
    seq_id_for_prompt = {}
    for i, prompt in enumerate(PROMPTS):
        input_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_lens.append(input_ids.shape[1])
        seq_id = scheduler.add_request(input_ids, MAX_NEW_TOKENS)
        seq_id_for_prompt[i] = seq_id

    results = scheduler.run_to_completion()

    for i, prompt in enumerate(PROMPTS):
        seq_id = seq_id_for_prompt[i]
        full_seq = results[seq_id]
        generated = full_seq[0, prompt_lens[i]:].tolist()
        assert generated == standalone_results[i], (
            f"Prompt {i} ('{prompt}') diverged when batched with others.\n"
            f"standalone: {loaded.tokenizer.decode(standalone_results[i])}\n"
            f"batched:    {loaded.tokenizer.decode(generated)}"
        )


def test_staggered_arrival_does_not_disturb_active_sequences():
    """A request added after the batch is already running (mid-stream
    admission) must not change the output of sequences already active --
    the actual scenario continuous batching exists for."""
    loaded = load_model(MODEL_NAME, device="cuda", attn_implementation="sdpa")

    standalone_results = [_run_standalone(loaded, p, MAX_NEW_TOKENS) for p in PROMPTS[:2]]

    manager = _build_manager(loaded.model.config)
    engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, manager, device="cuda")
    scheduler = ContinuousBatchingScheduler(engine, max_batch_size=4)

    input_ids_0 = loaded.tokenizer(PROMPTS[0], return_tensors="pt").input_ids.to("cuda")
    input_ids_1 = loaded.tokenizer(PROMPTS[1], return_tensors="pt").input_ids.to("cuda")
    seq_id_0 = scheduler.add_request(input_ids_0, MAX_NEW_TOKENS)

    # Run a few steps with only sequence 0 active before sequence 1 arrives.
    for _ in range(3):
        scheduler.step()

    seq_id_1 = scheduler.add_request(input_ids_1, MAX_NEW_TOKENS)

    results = scheduler.run_to_completion()

    gen_0 = results[seq_id_0][0, input_ids_0.shape[1]:].tolist()
    gen_1 = results[seq_id_1][0, input_ids_1.shape[1]:].tolist()

    assert gen_0 == standalone_results[0], "Sequence present from the start diverged after a later arrival joined"
    assert gen_1 == standalone_results[1], "Late-arriving sequence diverged from its standalone result"
