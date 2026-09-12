"""
Stage 3b: fused SiLU-and-multiply (the activation step of SwiGLU).

Qwen2.5's MLP computes down_proj(silu(gate_proj(x)) * up_proj(x)). The
silu(gate) * up part is unfused in PyTorch by default: silu(gate) writes
a full intermediate tensor to global memory, then a separate elementwise
multiply reads it back plus up, and writes the result again. For
Qwen2.5-1.5B's intermediate_size of 8960, that intermediate tensor is
sizeable and this happens once per layer per forward pass.

Fusing it into one kernel means the multiply happens in registers right
after computing silu, with one read of each input and one write of the
output, instead of an extra full round trip through global memory.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_fwd_kernel(
    Gate, Up, Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(Gate + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up + offs, mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    out = silu * up

    tl.store(Out + offs, out.to(Out.dtype.element_ty), mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused silu(gate) * up. Purely elementwise -- shapes must match
    exactly, any shape is fine since the kernel treats memory as flat."""
    assert gate.shape == up.shape
    assert gate.is_cuda and up.is_cuda

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)

    n_elements = gate_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _silu_mul_fwd_kernel[grid](gate_c, up_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(gate.shape)


def naive_silu_mul_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Unfused reference: two separate PyTorch ops, exactly what HF does
    by default."""
    return torch.nn.functional.silu(gate) * up
