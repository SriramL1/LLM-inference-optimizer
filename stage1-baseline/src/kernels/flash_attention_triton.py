"""
Stage 2: fused attention (FlashAttention-style), implemented in Triton.

Scope: this kernel targets PREFILL specifically -- self-attention where
query, key, and value all have the same sequence length. That's the
compute-bound phase Stage 2 is meant to optimize (see docs/stage1-baseline.md
for the prefill-vs-decode framing). Decode (1 new query token attending
over a growing KV-cache) stays on the Stage 1 eager path for now; it's a
different shape problem (memory-bandwidth-bound, tiny query block) that
Stage 4 (paged KV-cache) and Stage 6 (CUDA graphs) target instead.

Algorithm: block-tiled attention with online softmax (Dao et al.,
FlashAttention) -- never materializes the full (seq_len x seq_len)
attention matrix, so memory scales linearly instead of quadratically with
sequence length, and avoids a bandwidth-heavy round trip to global memory
for the intermediate softmax scores.

GQA note: HuggingFace models with grouped-query attention (fewer KV heads
than Q heads, e.g. Qwen2.5) are handled by repeating K/V heads up to the Q
head count *before* calling the kernel. That's the simple, obviously-correct
approach -- a more advanced kernel could broadcast KV heads internally and
skip the repeated memory, but that's left as a future optimization, not
Stage 2's job.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    H,                     # number of heads (for decomposing program_id(1))
    seq_len,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One program instance computes one BLOCK_M-sized slice of queries
    for one (batch, head) pair."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    Q += off_z * stride_qb + off_h * stride_qh
    K += off_z * stride_kb + off_h * stride_kh
    V += off_z * stride_vb + off_h * stride_vh
    Out += off_z * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Causal: a query block only ever needs to attend to key blocks up to
    # (and including) its own position -- skip the rest of the loop
    # entirely rather than masking wasted work.
    end_n = seq_len
    if CAUSAL:
        end_n = tl.minimum(seq_len, (start_m + 1) * BLOCK_M)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < seq_len, qk, float("-inf"))

        # --- online softmax rescaling ---
        # Rather than computing softmax over the full row at once (which
        # would require holding all seq_len scores in memory), we update a
        # running max (m_i) and running sum (l_i) block by block, rescaling
        # the accumulated output every time the running max changes.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    acc = acc / l_i[:, None]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """Fused causal self-attention.

    Args:
        q, k, v: (batch, num_heads, seq_len, head_dim). k/v may have fewer
            heads than q (GQA) -- they'll be repeat_interleave'd up to
            match. Assumes self-attention: q, k, v all share the same
            seq_len (this is the prefill case; decode is out of scope --
            see module docstring).
        causal: apply the standard causal (lower-triangular) mask.
        sm_scale: softmax scale; defaults to 1/sqrt(head_dim).

    Returns:
        (batch, num_heads, seq_len, head_dim) attention output, same dtype
        as q.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "flash_attention requires CUDA tensors"
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, kv_seq_len, kv_head_dim = k.shape
    assert kv_seq_len == seq_len, (
        "flash_attention (Stage 2) only implements self-attention (prefill), "
        f"got q seq_len={seq_len} but k seq_len={kv_seq_len}. Decode "
        "(cross-attention over a growing KV-cache) is intentionally out of "
        "scope -- see module docstring."
    )
    assert kv_head_dim == head_dim

    if num_kv_heads != num_q_heads:
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be a multiple of num_kv_heads (GQA)"
        rep = num_q_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    assert BLOCK_D == head_dim, (
        f"head_dim={head_dim} must be a power of 2 for this kernel "
        "(true for essentially all current LLM architectures, e.g. 64/128)"
    )

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_q_heads)

    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        num_q_heads,
        seq_len,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        CAUSAL=causal,
    )
    return o


def naive_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """Unfused PyTorch reference: materializes the full attention matrix.

    Used only for correctness-checking the Triton kernel -- this is
    deliberately the "obviously correct, obviously slow" version, the same
    role Stage 1's eager engine plays for the overall pipeline.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_kv_heads != num_q_heads:
        rep = num_q_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float())
    return out.to(q.dtype)
