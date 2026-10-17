"""
Stage 7 kernel and quantizer correctness checks.

Two different questions, deliberately tested separately:

1. Is the KERNEL correct? Checked against w8a16_reference, which
   dequantizes the same int8 weights to fp32 and does an ordinary fp32
   matmul. The kernel and this reference see identical quantized values,
   so any disagreement is a kernel bug -- it must match tightly,
   independent of how lossy quantization itself is.
2. How much accuracy does QUANTIZATION cost? Checked against the
   original fp16 nn.Linear. This is expected to differ (it's lossy by
   design) -- the test just bounds how much.

Shapes include Qwen2.5-1.5B's real projection sizes plus deliberately
awkward ones (K and N not multiples of any block size, M not a multiple
of BLOCK_M) so every masking path gets exercised -- the same reasoning as
the odd sequence lengths in the attention tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.kernels.w8a16_matmul_triton import w8a16_linear, w8a16_reference
from src.quantization.int8 import QuantizedLinearInt8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

# (K = in_features, N = out_features)
LAYER_SHAPES = [
    (1536, 1536),   # q_proj / o_proj
    (1536, 256),    # k_proj / v_proj (2 KV heads x 128)
    (1536, 8960),   # gate_proj / up_proj
    (8960, 1536),   # down_proj
    (100, 37),      # awkward: not a multiple of any block size
]

# Covers all three block-size tiers in _pick_config, plus non-multiples.
M_VALUES = [1, 7, 16, 100, 300]


def _make_linear(K, N, bias):
    torch.manual_seed(0)
    linear = torch.nn.Linear(K, N, bias=bias).cuda().half()
    with torch.no_grad():
        linear.weight.copy_(torch.randn(N, K, device="cuda") * 0.05)
        if bias:
            linear.bias.copy_(torch.randn(N, device="cuda") * 0.1)
    return linear


@pytest.mark.parametrize("K,N", LAYER_SHAPES)
@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("bias", [False, True])
def test_kernel_matches_dequantized_reference(K, N, M, bias):
    linear = _make_linear(K, N, bias)
    q = QuantizedLinearInt8.from_linear(linear)
    torch.manual_seed(1)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)

    kernel_out = w8a16_linear(x, q.weight_q, q.scale, q.bias)
    ref_out = w8a16_reference(x, q.weight_q, q.scale, q.bias)

    torch.testing.assert_close(kernel_out, ref_out, atol=2e-2, rtol=2e-2)


def test_kernel_handles_3d_input():
    """Real model activations are (batch, seq, hidden), not 2D."""
    linear = _make_linear(1536, 256, bias=True)
    q = QuantizedLinearInt8.from_linear(linear)
    x = torch.randn(2, 5, 1536, device="cuda", dtype=torch.float16)

    out = q(x)
    assert out.shape == (2, 5, 256)

    ref = w8a16_reference(x, q.weight_q, q.scale, q.bias)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("K,N", [(1536, 1536), (1536, 8960)])
def test_quantization_error_is_small_vs_fp16(K, N):
    """Bounds the accuracy cost of quantization itself (lossy by design).
    Per-channel int8 on roughly-Gaussian weights should land around ~1%
    relative error; the bound here is deliberately a bit loose."""
    linear = _make_linear(K, N, bias=False)
    q = QuantizedLinearInt8.from_linear(linear)
    torch.manual_seed(2)
    x = torch.randn(64, K, device="cuda", dtype=torch.float16)

    fp16_out = linear(x).float()
    int8_out = q(x).float()

    rel_err = (fp16_out - int8_out).norm() / fp16_out.norm()
    assert rel_err < 0.03, f"relative error {rel_err:.4f} higher than expected for per-channel int8"


def test_roundtrip_error_within_half_a_quantization_step():
    """Round-to-nearest means every dequantized weight is within
    scale/2 of the original -- a hard guarantee, not a statistical one."""
    linear = _make_linear(1536, 1536, bias=False)
    q = QuantizedLinearInt8.from_linear(linear)

    original = linear.weight.float()
    dequantized = q.dequantized_weight(torch.float32)
    max_allowed = (q.scale[:, None] / 2) * 1.001 + 1e-6  # tiny slack for fp rounding

    assert ((original - dequantized).abs() <= max_allowed).all()


def test_weight_bytes_roughly_halved():
    linear = _make_linear(1536, 8960, bias=False)
    q = QuantizedLinearInt8.from_linear(linear)
    fp16_bytes = linear.weight.numel() * linear.weight.element_size()
    assert q.num_bytes() < 0.55 * fp16_bytes
