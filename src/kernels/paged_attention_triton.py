"""
Stage 4: paged attention decode kernel.

Decode's attention shape: one new query token per sequence, attending
over that sequence's entire cached context -- which, under paging, lives
scattered across pages rather than one contiguous buffer. This kernel
reads the block table to find each page, gathers K/V from it, and runs
the same online-softmax accumulation flash attention uses, just over
page-sized chunks read via a runtime-computed (gathered) address instead
of a compile-time-known tiled address.

Scope: decode only (single query token). Prefill still uses Stage 2's
kernel or SDPA on the token span being processed for the very first
time -- paging only matters once tokens need to persist in a cache
across steps.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_decode_kernel(
    Q,                # (batch, num_q_heads, head_dim)
    K_cache,          # (num_pages, block_size, num_kv_heads, head_dim)
    V_cache,          # (num_pages, block_size, num_kv_heads, head_dim)
    block_tables,     # (batch, max_num_blocks) int32 -- physical page id per logical block
    context_lens,     # (batch,) int32 -- valid cached token count per sequence
    Out,              # (batch, num_q_heads, head_dim)
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kn, stride_kh, stride_kd,
    stride_vp, stride_vn, stride_vh, stride_vd,
    stride_btb, stride_btn,
    stride_ob, stride_oh, stride_od,
    num_q_heads,
    num_kv_heads,
    sm_scale,
    BLOCK_SIZE: tl.constexpr,       # tokens per page
    HEAD_DIM: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,   # compile-time upper bound on blocks/sequence
):
    """One program per (sequence, query head)."""
    batch_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    heads_per_group = num_q_heads // num_kv_heads
    kv_head_idx = q_head_idx // heads_per_group

    context_len = tl.load(context_lens + batch_idx)
    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + batch_idx * stride_qb + q_head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptr).to(tl.float32)  # (HEAD_DIM,)

    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_SIZE)

    for block_idx in range(0, MAX_NUM_BLOCKS):
        is_valid_block = block_idx < num_blocks

        bt_ptr = block_tables + batch_idx * stride_btb + block_idx * stride_btn
        page_id = tl.load(bt_ptr, mask=is_valid_block, other=0)

        token_pos_in_seq = block_idx * BLOCK_SIZE + offs_n
        valid_token_mask = is_valid_block & (token_pos_in_seq < context_len)

        k_ptrs = (
            K_cache + page_id * stride_kp + offs_n[:, None] * stride_kn
            + kv_head_idx * stride_kh + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=valid_token_mask[:, None], other=0.0).to(tl.float32)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale  # (BLOCK_SIZE,)
        qk = tl.where(valid_token_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v_ptrs = (
            V_cache + page_id * stride_vp + offs_n[:, None] * stride_vn
            + kv_head_idx * stride_vh + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=valid_token_mask[:, None], other=0.0).to(tl.float32)

        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    acc = acc / l_i

    out_ptr = Out + batch_idx * stride_ob + q_head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptr, acc.to(Out.dtype.element_ty))


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Args:
        q: (batch, num_q_heads, head_dim) -- single decode-step query.
        k_cache, v_cache: (num_pages, block_size, num_kv_heads, head_dim)
            -- the FULL physical page pool for one layer (not just this
            batch's pages; gathered via block_tables).
        block_tables: (batch, max_num_blocks) int32.
        context_lens: (batch,) int32.
        block_size: tokens per page (must match k_cache/v_cache's layout).

    Returns:
        (batch, num_q_heads, head_dim)
    """
    assert q.is_cuda
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    max_num_blocks = block_tables.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    out = torch.empty_like(q)
    grid = (batch, num_q_heads)

    _paged_attn_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, context_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_q_heads, num_kv_heads, sm_scale,
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim, MAX_NUM_BLOCKS=max_num_blocks,
    )
    return out


def naive_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """Ground-truth reference: for each sequence, gather its pages into a
    contiguous (context_len, num_kv_heads, head_dim) tensor in plain
    PyTorch, then run ordinary (unfused, unpaged) attention. Deliberately
    the "obviously correct, obviously slow" version the kernel above is
    checked against."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    heads_per_group = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    outputs = []
    for b in range(batch):
        ctx_len = context_lens[b].item()
        n_blocks = (ctx_len + block_size - 1) // block_size
        page_ids = block_tables[b, :n_blocks]

        k_gathered = k_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]
        v_gathered = v_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:ctx_len]

        # (ctx_len, num_kv_heads, head_dim) -> (num_kv_heads, ctx_len, head_dim)
        k_gathered = k_gathered.transpose(0, 1).float()
        v_gathered = v_gathered.transpose(0, 1).float()

        seq_out_heads = []
        for h in range(num_q_heads):
            kv_h = h // heads_per_group
            q_h = q[b, h].float()  # (head_dim,)
            scores = (q_h[None, :] * k_gathered[kv_h]).sum(-1) * sm_scale  # (ctx_len,)
            probs = torch.softmax(scores, dim=-1)
            out_h = (probs[:, None] * v_gathered[kv_h]).sum(0)  # (head_dim,)
            seq_out_heads.append(out_h)
        outputs.append(torch.stack(seq_out_heads, dim=0))

    return torch.stack(outputs, dim=0).to(q.dtype)
