"""
Stage 4: paged KV-cache memory manager.

The core idea (vLLM's PagedAttention): instead of one contiguous,
pre-allocated KV-cache tensor per sequence (wasteful -- reserves worst-
case length, and blocks efficient batching of variable-length
sequences), split the cache into fixed-size "pages" (like OS virtual
memory), allocate them on demand as a sequence grows, and keep a
per-sequence "block table" mapping logical token positions to physical
page indices. Memory is only used for tokens that actually exist.

Design: physical K/V storage is one big tensor per layer, shape
(max_pages, block_size, num_kv_heads, head_dim). Block tables are
shared across layers for the same sequence -- all layers process the
same tokens in lockstep, so "logical block 3 of sequence 7" refers to
the same physical page index in every layer's storage. Only the K/V
*values* differ per layer, not the allocation bookkeeping.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


class BlockAllocator:
    """Free-list based page allocator. Pages are just integer ids; the
    actual K/V storage lives in PagedKVCacheManager."""

    def __init__(self, num_pages: int):
        self.num_pages = num_pages
        self._free = list(range(num_pages))

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"Out of KV-cache pages (pool size={self.num_pages}). "
                "Increase max_pages or free unused sequences."
            )
        return self._free.pop()

    def free(self, page_id: int) -> None:
        self._free.append(page_id)

    @property
    def num_free(self) -> int:
        return len(self._free)


@dataclass
class BlockTable:
    """Tracks which physical pages hold a single sequence's tokens, in
    order, plus how many of the tokens in the cache are actually valid
    (the last page is usually only partially filled)."""
    page_ids: List[int] = field(default_factory=list)
    context_len: int = 0  # number of valid cached tokens

    def num_blocks(self) -> int:
        return len(self.page_ids)


class PagedKVCacheManager:
    """Owns the page pool, per-sequence block tables, and per-layer
    physical K/V storage.

    Usage per forward step, for a batch of sequences each appending
    num_new_tokens tokens (num_new_tokens is the same for all sequences
    in a batch in this simplified design -- prefill appends the whole
    prompt length at once, decode appends 1 token at a time):

        write_positions = manager.reserve(seq_id, num_new_tokens)
        # ... for each layer:
        manager.write(seq_id, layer_idx, k, v, write_positions)
        # ... after all layers, to run the attention kernel:
        block_table_tensor = manager.block_table_tensor(seq_id, max_blocks)
        context_len = manager.context_len(seq_id)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_pages: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_pages = max_pages
        self.device = device
        self.dtype = dtype

        self.allocator = BlockAllocator(max_pages)
        self.block_tables: Dict[int, BlockTable] = {}

        # One K and one V tensor per layer, each holding every page.
        cache_shape = (max_pages, block_size, num_kv_heads, head_dim)
        self.k_cache = [
            torch.zeros(cache_shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]

    def allocate_sequence(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            raise ValueError(f"seq_id {seq_id} already allocated")
        self.block_tables[seq_id] = BlockTable()

    def free_sequence(self, seq_id: int) -> None:
        table = self.block_tables.pop(seq_id)
        for page_id in table.page_ids:
            self.allocator.free(page_id)

    def context_len(self, seq_id: int) -> int:
        return self.block_tables[seq_id].context_len

    def reserve(self, seq_id: int, num_new_tokens: int) -> List[Tuple[int, int]]:
        """Allocates pages as needed for num_new_tokens new tokens,
        returns a list of (page_id, offset_in_page) for each new token in
        order. Call this ONCE per forward step (not once per layer) --
        the same reservation is reused for every layer's write() call,
        since allocation is layer-agnostic."""
        table = self.block_tables[seq_id]
        positions = []

        for _ in range(num_new_tokens):
            if table.context_len % self.block_size == 0:
                # current position starts a fresh page
                table.page_ids.append(self.allocator.allocate())
            page_idx_in_table = table.context_len // self.block_size
            offset = table.context_len % self.block_size
            page_id = table.page_ids[page_idx_in_table]
            positions.append((page_id, offset))
            table.context_len += 1

        return positions

    def write(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: List[Tuple[int, int]],
    ) -> None:
        """k, v: (num_new_tokens, num_kv_heads, head_dim). Writes each
        token's K/V into the page slot reserved for it by reserve()."""
        assert k.shape[0] == len(positions)
        for i, (page_id, offset) in enumerate(positions):
            self.k_cache[layer_idx][page_id, offset] = k[i]
            self.v_cache[layer_idx][page_id, offset] = v[i]

    def block_table_tensor(self, seq_id: int, max_blocks: int) -> torch.Tensor:
        """Returns a fixed-width (max_blocks,) int32 tensor of physical
        page ids for this sequence, padded with 0 beyond its actual
        block count (padding is safe -- the kernel masks by context_len,
        never reads padded entries)."""
        table = self.block_tables[seq_id]
        padded = table.page_ids + [0] * (max_blocks - len(table.page_ids))
        return torch.tensor(padded[:max_blocks], dtype=torch.int32, device=self.device)

    def batch_block_table_tensor(self, seq_ids: List[int], max_blocks: int) -> torch.Tensor:
        """(batch, max_blocks) int32 tensor, one row per sequence."""
        rows = [self.block_table_tensor(sid, max_blocks) for sid in seq_ids]
        return torch.stack(rows, dim=0)

    def batch_context_lens_tensor(self, seq_ids: List[int]) -> torch.Tensor:
        return torch.tensor(
            [self.context_len(sid) for sid in seq_ids], dtype=torch.int32, device=self.device
        )

    def memory_usage(self) -> dict:
        used_pages = self.max_pages - self.allocator.num_free
        bytes_per_page = (
            self.block_size * self.num_kv_heads * self.head_dim
            * 2  # K and V
            * self.num_layers
            * self.dtype.itemsize
        )
        return {
            "used_pages": used_pages,
            "free_pages": self.allocator.num_free,
            "total_pages": self.max_pages,
            "used_bytes": used_pages * bytes_per_page,
            "total_bytes": self.max_pages * bytes_per_page,
        }
