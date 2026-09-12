"""
Correctness checks for the Stage 3 kernels: fused RMSNorm and fused
silu-and-mul. Shapes include Qwen2.5-1.5B-Instruct's actual dimensions
(hidden_size=1536, intermediate_size=8960) plus a deliberately
non-power-of-2 hidden size to exercise the RMSNorm kernel's masking path
-- the same reasoning as the odd sequence length in
test_flash_attention_kernel.py: boundary conditions are where tiled/
masked kernels most often hide bugs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.kernels.fused_rmsnorm_triton import fused_rmsnorm, naive_rmsnorm_reference
from src.kernels.fused_swiglu_triton import fused_silu_mul, naive_silu_mul_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

# (batch, seq_len, hidden_size)
RMSNORM_SHAPES = [
    (1, 8, 1536),     # Qwen2.5-1.5B-Instruct hidden_size
    (2, 256, 1536),
    (1, 8, 100),      # non-power-of-2 hidden size -- exercises masking
]

# (batch, seq_len, intermediate_size)
SWIGLU_SHAPES = [
    (1, 8, 8960),     # Qwen2.5-1.5B-Instruct intermediate_size
    (2, 256, 8960),
    (1, 8, 37),       # arbitrary odd size -- exercises masking
]


@pytest.mark.parametrize("batch,seq_len,hidden_size", RMSNORM_SHAPES)
def test_rmsnorm_matches_reference(batch, seq_len, hidden_size):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16

    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    fused_out = fused_rmsnorm(x, weight, eps=1e-6)
    ref_out = naive_rmsnorm_reference(x, weight, eps=1e-6)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch,seq_len,intermediate_size", SWIGLU_SHAPES)
def test_silu_mul_matches_reference(batch, seq_len, intermediate_size):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16

    gate = torch.randn(batch, seq_len, intermediate_size, device=device, dtype=dtype)
    up = torch.randn(batch, seq_len, intermediate_size, device=device, dtype=dtype)

    fused_out = fused_silu_mul(gate, up)
    ref_out = naive_silu_mul_reference(gate, up)

    torch.testing.assert_close(fused_out, ref_out, atol=2e-2, rtol=2e-2)
