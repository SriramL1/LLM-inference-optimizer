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
) -> LoadedModel:
    """Load a causal LM + tokenizer for baseline benchmarking.

    Defaults to a small (1.5B) modern-architecture model (RMSNorm, RoPE,
    grouped-query attention) so it comfortably fits an RTX 4070 (12GB) and
    is architecturally representative of what later optimization stages
    (fused attention, paged KV-cache, quantization) will target.

    attn_implementation="eager" is deliberate: Stage 1 is the unoptimized
    reference. PyTorch's SDPA / FlashAttention backends get introduced
    later as an explicit optimization stage, not baked into the baseline.
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
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, device=device, dtype=dtype)
