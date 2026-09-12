"""
Model loading utilities for the baseline inference engine.
"""
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: any
    device: str
    dtype: torch.dtype


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    attn_implementation: str = "eager",
) -> LoadedModel:
    """Load a causal LM + tokenizer for benchmarking.

    Defaults to a small (1.5B) modern-architecture model (RMSNorm, RoPE,
    grouped-query attention) so it comfortably fits an RTX 4070 (12GB) and
    is architecturally representative of what later optimization stages
    (fused attention, paged KV-cache, quantization) will target.

    attn_implementation defaults to "eager": Stage 1's unoptimized
    reference. Pass "sdpa" for PyTorch's built-in fused attention, or the
    name a custom implementation was registered under via
    transformers.AttentionInterface.register(...) (see
    src/engine/flash_attention_patch.py for Stage 2's Triton kernel).
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This baseline harness targets a CUDA GPU "
            "(e.g. an RTX 4070). Install a CUDA-enabled PyTorch build, or "
            "pass device='cpu' for a much slower correctness-only run."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, device=device, dtype=dtype)
