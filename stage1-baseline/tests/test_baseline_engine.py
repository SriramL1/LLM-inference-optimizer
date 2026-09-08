"""
Correctness check: the manual prefill/decode loop must produce identical
greedy-decoded token ids to HuggingFace's model.generate(). This is the
regression test every later stage must also pass -- optimizations should
change speed and memory, never the output tokens (for greedy decoding).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.engine.model_loader import load_model
from src.engine.baseline_engine import BaselineEngine

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_manual_loop_matches_hf_generate():
    loaded = load_model(MODEL_NAME, device="cuda")
    engine = BaselineEngine(loaded.model, loaded.tokenizer, device="cuda")

    prompt = "The capital of France is"
    input_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    max_new_tokens = 16

    ref_output = loaded.model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    ref_new_tokens = ref_output[0, input_ids.shape[1]:].tolist()

    past_key_values, next_token = engine.prefill(input_ids)
    manual_tokens = [next_token.item()]
    cur = next_token
    for _ in range(max_new_tokens - 1):
        past_key_values, cur = engine.decode_step(cur, past_key_values)
        manual_tokens.append(cur.item())

    assert manual_tokens == ref_new_tokens, (
        f"Manual loop diverged from HF generate.\n"
        f"manual: {manual_tokens}\nref:    {ref_new_tokens}"
    )
