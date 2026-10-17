"""
Stage 7: weight-only INT8 matmul (W8A16).

Why this helps: decode is memory-bandwidth-bound (Stage 1). Every decode
step re-reads every weight matrix from GPU memory to produce one token,
so the time is dominated by *bytes of weights read*, not arithmetic.
INT8 weights are half the bytes of fp16, so the weight traffic -- the
dominant cost -- should roughly halve. Activations stay fp16 (hence
"W8A16"), which is what keeps this simple and accurate: no activation
quantization, no calibration data.

Scheme: per-output-channel symmetric quantization. For a weight matrix
W of shape (N, K) (N output features, K input features, nn.Linear's
layout), each row n gets one scale[n] = max|W[n, :]| / 127, and
Wq[n, k] = round(W[n, k] / scale[n]) in [-127, 127].

Kernel trick: because scale is constant along the reduction axis K,
    y[m, n] = sum_k x[m, k] * (Wq[n, k] * scale[n])
            = scale[n] * sum_k x[m, k] * Wq[n, k]
so the kernel accumulates x @ Wq^T in fp32 using the raw integer values
(converted to fp16 -- every int8 is exactly representable) and multiplies
by scale ONCE at the end, instead of dequantizing every weight tile.
Mathematically identical, strictly less work.

Deliberately NO @triton.autotune: autotuning runs benchmarking launches
on first call, which is incompatible with CUDA graph capture (Stage 6) --
a graphed engine's first call IS the capture. Block sizes are instead
picked by a small, fixed heuristic on M (see _pick_config). That means
this kernel is NOT tuned for any specific GPU -- the numbers it produces
are a first-pass result, and the isolated kernel benchmark exists
precisely to show how competitive it is against cuBLAS before anyone
reads an end-to-end number.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_matmul_kernel(
    X, Wq, Scale, Bias, Out,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """out[M, N] = (X[M, K] @ Wq[N, K]^T) * Scale[N] (+ Bias[N])."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Loaded directly as (BLOCK_K, BLOCK_N) -- i.e. already transposed --
    # so it can feed tl.dot without an explicit transpose.
    w_ptrs = Wq + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0)
        acc += tl.dot(x, w.to(x.dtype))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    scale = tl.load(Scale + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _pick_config(M: int) -> dict:
    """Fixed, shape-based block-size heuristic -- no autotuning (see
    module docstring). Decode (small M) wants many narrow programs so the
    GPU has enough parallel memory requests in flight; prefill (large M)
    wants bigger tiles for arithmetic reuse. These are reasonable
    starting points, not tuned values."""
    if M <= 16:
        return dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, num_warps=4, num_stages=3)
    if M <= 256:
        return dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)


def w8a16_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """x: (..., K) fp16/bf16. w_q: (N, K) int8. scale: (N,) float32.
    bias: optional (N,). Returns (..., N) in x's dtype."""
    assert x.is_cuda and w_q.is_cuda
    assert w_q.dtype == torch.int8, f"w_q must be int8, got {w_q.dtype}"
    N, K = w_q.shape
    assert x.shape[-1] == K, f"x last dim {x.shape[-1]} != weight in_features {K}"

    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    cfg = _pick_config(M)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))

    _w8a16_matmul_kernel[grid](
        x2, w_q, scale,
        bias if bias is not None else scale,  # dummy pointer, never read when HAS_BIAS=False
        out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        w_q.stride(0), w_q.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out.reshape(*x.shape[:-1], N)


def w8a16_reference(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Obviously-correct reference: dequantize the whole weight matrix to
    fp32 first, then an ordinary fp32 linear. Isolates *kernel*
    correctness from *quantization* error -- the kernel must match this
    tightly, independent of how much accuracy quantization itself costs."""
    w = w_q.float() * scale.float()[:, None]
    b = bias.float() if bias is not None else None
    return torch.nn.functional.linear(x.float(), w, b).to(x.dtype)
