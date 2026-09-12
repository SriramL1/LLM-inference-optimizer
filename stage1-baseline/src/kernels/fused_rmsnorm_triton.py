"""
Stage 3a: fused RMSNorm.

HF's RMSNorm (Qwen2RMSNorm and equivalents across the Llama family) is
implemented as a chain of separate PyTorch ops -- pow, mean, rsqrt, two
multiplies -- each its own CUDA kernel launch with a round trip to global
memory in between. RMSNorm runs twice per transformer layer (before
attention, before the MLP), so for a 28-layer model that's dozens of tiny
kernel launches per forward pass, each dominated by launch overhead and
memory traffic rather than actual compute -- normalization itself is
cheap; the *n* separate reads/writes are what cost time.

This fuses the whole thing into one kernel: one program instance per row
(one token's hidden vector), the entire reduction (sum of squares) and
elementwise rescale done in a single pass over that row, one read and one
write of the row instead of five.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Out,
    stride_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. BLOCK_SIZE must be >= N (the whole hidden
    dimension is normalized together, so it has to fit in one block --
    true for any current LLM hidden size, which is always well under
    Triton's practical block-size ceiling)."""
    row = tl.program_id(0)
    X += row * stride_row
    Out += row * stride_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Out + cols, y.to(Out.dtype.element_ty), mask=mask)


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in replacement for HF's RMSNorm.forward. x: (..., hidden_size),
    weight: (hidden_size,). Normalizes over the last dimension."""
    assert x.is_cuda and weight.is_cuda
    orig_shape = x.shape
    N = orig_shape[-1]
    assert weight.shape == (N,)

    x_flat = x.reshape(-1, N).contiguous()
    M = x_flat.shape[0]
    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x_flat, weight, out,
        x_flat.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape)


def naive_rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unfused reference, matching HF's actual RMSNorm formula exactly
    (compute in fp32, cast back before the weight multiply -- this cast
    order matters for numerical match, not just the math)."""
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)
