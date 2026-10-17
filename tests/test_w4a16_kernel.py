"""
Stage 7b kernel and quantizer correctness checks (INT4, group-wise).

Same two-question structure as Stage 7:
1. Is the KERNEL (and the nibble packing) correct? Checked against
   w4a16_reference, which unpacks and dequantizes the SAME packed weights
   to fp32 and does an ordinary matmul -- any disagreement is a kernel or
   packing bug, independent of how lossy 4-bit is.
2. How much accuracy does QUANTIZATION cost? Checked against the original
   fp16 nn.Linear. Lossy by design; the test only bounds it.

K must be a multiple of 128 (group size). Shapes are Qwen2.5-1.5B's real
projections plus an awkward one (K=384, N=100: N not a multiple of any
block size, so the N-masking path runs). M values cover all three
block-size tiers in _pick_config plus non-multiples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.kernels.w4a16_matmul_triton import (
    GROUP_SIZE, pack_int4, unpack_int4, w4a16_linear, w4a16_reference,
)
from src.quantization.int4 import QuantizedLinearInt4

LAYER_SHAPES = [
    (1536, 1536),
    (1536, 256),
    (1536, 8960),
    (8960, 1536),
    (384, 100),
]
M_VALUES = [1, 7, 16, 100, 300]


def _make_linear(K, N, bias):
    torch.manual_seed(0)
    linear = torch.nn.Linear(K, N, bias=bias).cuda().half()
    with torch.no_grad():
        linear.weight.copy_(torch.randn(N, K, device="cuda") * 0.05)
        if bias:
            linear.bias.copy_(torch.randn(N, device="cuda") * 0.1)
    return linear


def test_pack_unpack_roundtrip():
    """Pure bit-twiddling, no kernel: every value 0..15 must survive."""
    torch.manual_seed(0)
    q = torch.randint(0, 16, (37, 384), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)
    # Every nibble value in both positions.
    q2 = torch.arange(16, dtype=torch.uint8).repeat(2, 8)  # (2, 128)
    assert torch.equal(unpack_int4(pack_int4(q2)), q2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("K,N", LAYER_SHAPES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("bias", [False, True])
def test_kernel_matches_dequantized_reference(K, N, M, bias):
    linear = _make_linear(K, N, bias)
    q = QuantizedLinearInt4.from_linear(linear)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)

    out = w4a16_linear(x, q.weight_packed, q.scales, q.bias)
    ref = w4a16_reference(x, q.weight_packed, q.scales, q.bias)

    assert out.shape == (M, N) and out.dtype == torch.float16
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_leading_dims_preserved():
    linear = _make_linear(1536, 256, bias=False)
    q = QuantizedLinearInt4.from_linear(linear)
    x = torch.randn(2, 5, 1536, device="cuda", dtype=torch.float16)
    out = q(x)
    assert out.shape == (2, 5, 256)
    ref = w4a16_reference(x.reshape(-1, 1536), q.weight_packed, q.scales).reshape(2, 5, 256)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_quantization_error_bounded_vs_fp16():
    """4-bit is lossy: with max-abs group scales on Gaussian weights the
    expected relative output error is roughly 10-15%. The bound is
    deliberately loose; it exists to catch a broken quantizer (sign
    flips, wrong scale), not to certify accuracy."""
    linear = _make_linear(1536, 1536, bias=False)
    q = QuantizedLinearInt4.from_linear(linear)
    x = torch.randn(64, 1536, device="cuda", dtype=torch.float16)

    exact = torch.nn.functional.linear(x, linear.weight).float()
    approx = q(x).float()
    rel = (approx - exact).norm() / exact.norm()
    assert rel.item() < 0.2, f"relative output error {rel.item():.3f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_weight_roundtrip_within_half_step():
    """Hard guarantee for round-to-nearest: no dequantized weight is more
    than half a quantization step from the original. 0.51 (not 0.5)
    allows for scales being stored in fp16, which can shift the clamp
    boundary by a hair."""
    linear = _make_linear(1536, 300, bias=False)
    q = QuantizedLinearInt4.from_linear(linear)
    dq = q.dequantized_weight(torch.float32)
    step = q.scales.float().t().repeat_interleave(GROUP_SIZE, dim=1)  # (N, K)
    err = (dq - linear.weight.float()).abs()
    assert (err <= 0.51 * step + 1e-6).all(), f"max err/step = {(err/step).max().item():.3f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_weight_bytes_about_a_quarter_of_fp16():
    linear = _make_linear(1536, 8960, bias=False)
    q = QuantizedLinearInt4.from_linear(linear)
    fp16_bytes = linear.weight.numel() * linear.weight.element_size()
    assert q.num_bytes() < 0.28 * fp16_bytes, f"ratio {q.num_bytes() / fp16_bytes:.3f}"
