"""
Stage 7b: weight-only INT4 matmul (W4A16), group-wise symmetric.

Same thesis as INT8 (Stage 7): decode is memory-bound, so bytes of
weights read set the step time. INT4 is a quarter of fp16's bytes
(~0.26x including scales), so the ceiling is higher -- but 4 bits is
coarse enough that ONE scale per output row (what INT8 used) wrecks
accuracy. Instead each row is split into groups of GROUP_SIZE=128
consecutive input features, and each group gets its own scale:
    scale[g, n] = max|W[n, g*128:(g+1)*128]| / 7
    q = round(W / scale) in [-7, 7], stored as q + 8 in [1, 15]
(Offset-binary keeps nibbles unsigned, so unpacking is just & and >>.)

Packing layout (chosen so the kernel's activation loads stay contiguous):
weights are packed two-per-byte along K, but NOT adjacent pairs. Within
each group of 128, byte j (0..63) holds element j in its LOW nibble and
element j+64 in its HIGH nibble. So one group is 64 contiguous bytes,
and the kernel multiplies x[:, g*128 : g*128+64] by the low nibbles and
x[:, g*128+64 : g*128+128] by the high nibbles -- two contiguous x
loads, no strided/interleaved access.

Kernel structure: BLOCK_K == GROUP_SIZE, so each loop iteration covers
exactly one group and needs exactly one scale per output column:
    part = x_lo @ W_lo + x_hi @ W_hi          (fp32, raw integers)
    acc += part * scale[g, :]
Scale cannot be hoisted to the end here (it varies along K, unlike
INT8's per-row scale), so it is applied once per group instead.

Like the INT8 kernel: NO @triton.autotune (incompatible with CUDA graph
capture), fixed shape-based block sizes, therefore untuned. Requires
K % 128 == 0 (true for all Qwen2.5-1.5B projections: 1536 and 8960).
"""
import torch
import triton
import triton.language as tl

GROUP_SIZE = 128
HALF = GROUP_SIZE // 2


# ---------------------------------------------------------------- packing
def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: (N, K) integer tensor with values in [0, 15] (offset-binary),
    K % 128 == 0. Returns (N, K//2) uint8 in the layout described above."""
    N, K = q.shape
    assert K % GROUP_SIZE == 0, f"K={K} must be a multiple of {GROUP_SIZE}"
    g = q.to(torch.uint8).reshape(N, K // GROUP_SIZE, 2, HALF)
    packed = g[:, :, 0, :] | (g[:, :, 1, :] << 4)
    return packed.reshape(N, K // 2).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: (N, K//2) uint8 -> (N, K) uint8 in [0, 15]."""
    N, half_k = packed.shape
    p = packed.reshape(N, half_k // HALF, HALF)
    lo = p & 0xF
    hi = p >> 4
    return torch.stack([lo, hi], dim=2).reshape(N, half_k * 2)


# ----------------------------------------------------------------- kernel
@triton.jit
def _w4a16_matmul_kernel(
    X, Wp, Scale, Bias, Out,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sg,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,
):
    """out[M, N] = sum over groups g of (X_g @ dequant(Wp_g)^T) (+ Bias[N])."""
    HALF_G: tl.constexpr = GROUP // 2
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, HALF_G)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n < N

    x_lo_ptrs = X + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xk
    x_hi_ptrs = x_lo_ptrs + HALF_G * stride_xk
    # (HALF_G, BLOCK_N): already "transposed" for tl.dot.
    w_ptrs = Wp + offs_n[None, :] * stride_wn + offs_h[:, None] * stride_wk
    s_ptrs = Scale + offs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // GROUP):
        x_lo = tl.load(x_lo_ptrs, mask=m_mask, other=0.0)
        x_hi = tl.load(x_hi_ptrs, mask=m_mask, other=0.0)
        packed = tl.load(w_ptrs, mask=n_mask[None, :], other=0).to(tl.int32)
        w_lo = ((packed & 0xF) - 8).to(x_lo.dtype)
        w_hi = ((packed >> 4) - 8).to(x_lo.dtype)

        part = tl.dot(x_lo, w_lo)
        part = tl.dot(x_hi, w_hi, part)

        scale = tl.load(s_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        acc += part * scale[None, :]

        x_lo_ptrs += GROUP * stride_xk
        x_hi_ptrs += GROUP * stride_xk
        w_ptrs += HALF_G * stride_wk
        s_ptrs += stride_sg

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask & n_mask[None, :])


def _pick_config(M: int, N: int) -> dict:
    """Fixed heuristic, no autotune (see module docstring). Untuned.
    Decode-sized M: narrow N tiles so even skinny layers (N=256) launch
    enough programs to keep the GPU busy -- the INT8 benchmark showed
    small layers losing to cuBLAS, a guess at why is too few programs.
    num_stages=2 for large tiles to stay under the ~99KB shared-memory
    limit of consumer Ada GPUs."""
    if M <= 16:
        return dict(BLOCK_M=16, BLOCK_N=16 if N <= 2048 else 32, num_warps=4, num_stages=3)
    if M <= 256:
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2)


def w4a16_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """x: (..., K) fp16/bf16. w_packed: (N, K//2) uint8. scales: (K//128, N)
    fp16. bias: optional (N,). Returns (..., N) in x's dtype."""
    assert x.is_cuda and w_packed.is_cuda
    assert w_packed.dtype == torch.uint8, f"w_packed must be uint8, got {w_packed.dtype}"
    N, half_k = w_packed.shape
    K = half_k * 2
    assert x.shape[-1] == K, f"x last dim {x.shape[-1]} != weight in_features {K}"
    assert K % GROUP_SIZE == 0
    assert scales.shape == (K // GROUP_SIZE, N), f"scales shape {tuple(scales.shape)}"

    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    cfg = _pick_config(M, N)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))

    _w4a16_matmul_kernel[grid](
        x2, w_packed, scales,
        bias if bias is not None else scales,  # dummy pointer, never read when HAS_BIAS=False
        out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], GROUP=GROUP_SIZE,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out.reshape(*x.shape[:-1], N)


def dequantize_int4(w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(N, K//2) uint8 + (K//128, N) scales -> (N, K) float32 weights."""
    q = unpack_int4(w_packed).float() - 8.0                    # (N, K)
    N, K = q.shape
    s = scales.float().t().reshape(N, K // GROUP_SIZE, 1)       # (N, G, 1)
    return (q.reshape(N, K // GROUP_SIZE, GROUP_SIZE) * s).reshape(N, K)


def w4a16_reference(x, w_packed, scales, bias=None):
    """Obviously-correct reference: dequantize everything to fp32, then an
    ordinary fp32 linear. Kernel must match this tightly; any gap is a
    kernel (or packing) bug, independent of how lossy 4-bit is."""
    w = dequantize_int4(w_packed, scales)
    b = bias.float() if bias is not None else None
    return torch.nn.functional.linear(x.float(), w, b).to(x.dtype)
