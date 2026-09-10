"""
Stage 2b: wire the Stage 2 Triton kernel into the real model.

Uses HuggingFace's documented AttentionInterface registration mechanism
(https://huggingface.co/docs/transformers/attention_interface) rather than
monkey-patching model internals directly -- the model calls our function
once per attention layer per forward pass, exactly like it would call its
built-in eager/sdpa implementations.

Dispatch logic, not a blanket replacement:
Our kernel (src/kernels/flash_attention_triton.py) only implements
self-attention with equal query/key length and no padding -- that's
prefill. Decode calls attention with a 1-token query against a much
longer KV-cache: a different shape entirely, out of the kernel's scope.
So every call is inspected at runtime and routed to the Triton kernel
only when it actually matches prefill's shape; everything else
(decode steps, padded batches) falls back to PyTorch's own SDPA
attention. This means enabling this attention implementation is safe by
construction -- it never silently produces wrong output outside the
kernel's tested scope, it just doesn't speed that case up.
"""
from typing import Optional

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from src.kernels.flash_attention_triton import flash_attention

ATTN_IMPL_NAME = "triton_attn"


def _is_unpadded(attention_mask: Optional[torch.Tensor]) -> bool:
    """True if there's no per-sequence padding to worry about.

    Our kernel implements a pure causal mask with no padding-mask support,
    so any batch with real padding must fall back to SDPA. This project's
    own benchmarks never pad (every sequence in a batch is the same
    prompt length), so in practice this fallback is rarely hit here -- but
    a real deployment with mixed-length batches would hit it constantly,
    which is exactly the gap Stage 5 (continuous batching) exists to
    close properly, not something to hack around here.
    """
    if attention_mask is None:
        return True
    if attention_mask.dtype == torch.bool:
        return bool(attention_mask.all())
    # additive mask convention: 0 = unmasked, large negative = masked
    return bool((attention_mask == 0).all())


def triton_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
):
    """Per-call dispatch: Stage 2 Triton kernel for prefill, SDPA for
    everything else. Matches the (attn_output, attn_weights) return
    convention every AttentionInterface implementation in transformers
    follows, so this is a drop-in swap for "eager" or "sdpa"."""
    q_len = query.shape[-2]
    kv_len = key.shape[-2]

    is_prefill_self_attention = q_len == kv_len and q_len > 1
    can_use_triton = (
        is_prefill_self_attention
        and query.is_cuda
        and _is_unpadded(attention_mask)
    )

    if can_use_triton:
        attn_output = flash_attention(query, key, value, causal=True, sm_scale=scaling)
        # (batch, heads, seq, dim) -> (batch, seq, heads, dim), matching
        # what sdpa_attention_forward returns -- the caller reshapes this
        # to (batch, seq, hidden) itself.
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, **kwargs,
    )


def register_triton_flash_attention():
    """Call once, before loading a model, to make ATTN_IMPL_NAME available
    as an attn_implementation choice (e.g. load_model(..., attn_implementation=ATTN_IMPL_NAME))."""
    AttentionInterface.register(ATTN_IMPL_NAME, triton_flash_attention_forward)
