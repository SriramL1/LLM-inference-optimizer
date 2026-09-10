"""
Correctness check for the Stage 2 Triton flash-attention kernel.

Compares against two references:
  1. naive_attention_reference -- unfused PyTorch (materializes full
     attention matrix). This is the ground-truth check: if the Triton
     kernel disagrees with this by more than fp16 tolerance, the kernel
     is wrong, full stop.
  2. torch.nn.functional.scaled_dot_product_attention -- PyTorch's own
     fused/flash backend. Not the correctness ground truth (it's also an
     optimized implementation, not a naive reference) but a useful sanity
     check and, in the benchmark script, the actual competitor we're
     measuring our kernel against.

Shapes mirror Qwen2.5-1.5B-Instruct's actual config (12 query heads, 2 KV
heads i.e. GQA with group size 6, head_dim 128) so a pass here means the
kernel is ready to plug into the real model, not just a toy shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.kernels.flash_attention_triton import flash_attention, naive_attention_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

# (batch, num_q_heads, num_kv_heads, seq_len, head_dim)
SHAPES = [
    (1, 12, 2, 64, 128),     # Qwen2.5-1.5B-Instruct-like, short seq
    (2, 12, 2, 256, 128),    # Qwen2.5-1.5B-Instruct-like, longer seq
    (1, 8, 8, 128, 64),      # no GQA (num_kv_heads == num_q_heads), different head_dim
    (1, 12, 2, 65, 128),     # seq_len NOT a multiple of BLOCK_M/BLOCK_N -- exercises masking
]


@pytest.mark.parametrize("batch,num_q_heads,num_kv_heads,seq_len,head_dim", SHAPES)
def test_matches_naive_reference(batch, num_q_heads, num_kv_heads, seq_len, head_dim):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1

    triton_out = flash_attention(q, k, v, causal=True)
    ref_out = naive_attention_reference(q, k, v, causal=True)

    torch.testing.assert_close(triton_out, ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch,num_q_heads,num_kv_heads,seq_len,head_dim", SHAPES)
def test_matches_sdpa(batch, num_q_heads, num_kv_heads, seq_len, head_dim):
    """Sanity check against PyTorch's own fused attention. Also serves as
    a second independent implementation to cross-check against, in case
    the naive reference and the Triton kernel happened to share a bug."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1

    triton_out = flash_attention(q, k, v, causal=True)

    if num_kv_heads != num_q_heads:
        rep = num_q_heads // num_kv_heads
        k_sdpa = k.repeat_interleave(rep, dim=1)
        v_sdpa = v.repeat_interleave(rep, dim=1)
    else:
        k_sdpa, v_sdpa = k, v

    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q, k_sdpa, v_sdpa, is_causal=True
    )

    torch.testing.assert_close(triton_out, sdpa_out, atol=2e-2, rtol=2e-2)


def test_non_causal():
    """Non-causal path (full bidirectional attention) -- not what the LLM
    engine uses, but exercised here since it's a real code path in the
    kernel (CAUSAL=False) and should also be correct."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    batch, num_heads, seq_len, head_dim = 2, 4, 128, 64

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1

    triton_out = flash_attention(q, k, v, causal=False)
    ref_out = naive_attention_reference(q, k, v, causal=False)

    torch.testing.assert_close(triton_out, ref_out, atol=2e-2, rtol=2e-2)
