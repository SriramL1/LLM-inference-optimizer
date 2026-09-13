"""
Correctness checks for the Stage 4 paged-attention decode kernel and the
memory manager it depends on.

The variable-length-batch test matters most here: paging's entire reason
for existing is letting sequences of different lengths share a page pool
efficiently, so a test that only exercises uniform-length batches would
miss the actual point of this stage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

triton = pytest.importorskip("triton", reason="triton not installed")

from src.kv_cache.block_manager import PagedKVCacheManager
from src.kernels.paged_attention_triton import (
    paged_attention_decode,
    naive_paged_attention_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

BLOCK_SIZE = 16
HEAD_DIM = 128
NUM_KV_HEADS = 2
NUM_Q_HEADS = 12  # Qwen2.5-1.5B-Instruct: GQA group size 6


def _build_manager(context_lens, block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM):
    """Builds a manager with num_layers=1, fills each sequence's cache
    with random K/V up to its context_len, returns (manager, seq_ids)."""
    torch.manual_seed(0)
    manager = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_pages=256, device="cuda", dtype=torch.float16,
    )
    seq_ids = list(range(len(context_lens)))
    for seq_id, ctx_len in zip(seq_ids, context_lens):
        manager.allocate_sequence(seq_id)
        k = torch.randn(ctx_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1
        v = torch.randn(ctx_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1
        positions = manager.reserve(seq_id, ctx_len)
        manager.write(seq_id, layer_idx=0, k=k, v=v, positions=positions)
    return manager, seq_ids


@pytest.mark.parametrize("context_lens", [
    [16],            # exactly one full page
    [17],            # one full page + 1 token in a second page
    [5],             # partial single page, less than block_size
    [16, 33, 100],   # variable-length batch -- the actual point of paging
])
def test_paged_attention_matches_naive(context_lens):
    manager, seq_ids = _build_manager(context_lens)
    max_blocks = max((c + BLOCK_SIZE - 1) // BLOCK_SIZE for c in context_lens)

    batch = len(context_lens)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1

    block_tables = manager.batch_block_table_tensor(seq_ids, max_blocks)
    ctx_lens_tensor = manager.batch_context_lens_tensor(seq_ids)

    kernel_out = paged_attention_decode(
        q, manager.k_cache[0], manager.v_cache[0], block_tables, ctx_lens_tensor, BLOCK_SIZE
    )
    ref_out = naive_paged_attention_reference(
        q, manager.k_cache[0], manager.v_cache[0], block_tables, ctx_lens_tensor, BLOCK_SIZE
    )

    torch.testing.assert_close(kernel_out, ref_out, atol=2e-2, rtol=2e-2)


def test_manager_allocates_and_frees_pages_correctly():
    manager = PagedKVCacheManager(
        num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE, max_pages=10, device="cuda", dtype=torch.float16,
    )
    manager.allocate_sequence(0)
    assert manager.allocator.num_free == 10

    # 17 tokens at block_size=16 needs 2 pages
    manager.reserve(0, 17)
    assert manager.allocator.num_free == 8
    assert manager.context_len(0) == 17

    manager.free_sequence(0)
    assert manager.allocator.num_free == 10


def test_manager_raises_when_pool_exhausted():
    manager = PagedKVCacheManager(
        num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE, max_pages=1, device="cuda", dtype=torch.float16,
    )
    manager.allocate_sequence(0)
    manager.reserve(0, BLOCK_SIZE)  # exactly fills the one page

    with pytest.raises(RuntimeError, match="Out of KV-cache pages"):
        manager.reserve(0, 1)  # needs a second page -- none left
