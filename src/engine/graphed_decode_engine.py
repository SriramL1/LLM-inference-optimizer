"""
Stage 6: CUDA graph capture for decode.

Decode's cost problem that fused kernels (Stage 2/3) and paging (Stage
4) don't address: each decode step launches dozens of tiny CUDA kernels
(per layer: q/k/v proj, RoPE, attention, o_proj, 2 norms, MLP's
matmuls) to process exactly ONE new token. At this scale, Python
dispatch and CUDA kernel *launch* overhead -- not the kernels' actual
GPU execution time -- can dominate wall-clock time. CUDA graphs capture
the entire sequence of kernel launches once, then replay the whole
thing with a single API call, eliminating per-step Python/launch
overhead for every call after the first capture.

Hard constraint: a captured graph's kernel launches reference FIXED
memory addresses and FIXED tensor shapes -- nothing about that sequence
can change between replays. This is incompatible with naively reusing
PagedEngine.decode_step as written: it builds a fresh, varying-size
block_table tensor and calls torch.tensor(...) fresh every call, each
allocating new memory. Capturing that literally would freeze the graph
to whatever one sequence's one specific context_len happened to be at
capture time.

The fix: pre-allocate static buffers ONCE, sized for MAX_CONTEXT_LEN.
Every decode call -- for any sequence, at any real context length --
overwrites those same buffers' *contents* (via .copy_()/.fill_(), which
are themselves ordinary graph-capturable ops) rather than constructing
new tensors, then replays the SAME captured graph. This works because
paged_attention_decode already masks by the actual context_len at
runtime (a Stage 4 design choice originally motivated by variable-length
batching, which turns out to be exactly what graph capture needs too):
a graph captured with a fixed MAX_NUM_BLOCKS safely handles any real
context_len up to that maximum, paying only for some masked-out (wasted)
iterations at shorter lengths -- the same tradeoff Stage 4 already
documented and accepted.

Scope: single-sequence decode, reused across sequences by overwriting
the same static buffers -- NOT combined with Stage 5's continuous
batching. A dynamically-changing batch composition (sequence count
varying every iteration) is a substantially harder graph-capture
problem; real systems handle it by capturing several graphs for a few
fixed batch-size "buckets" and dispatching to whichever fits. Out of
scope for this pass.
"""
import torch

from src.engine.paged_engine import PagedEngine
from src.kernels.paged_attention_triton import paged_attention_decode


class GraphedDecodeEngine(PagedEngine):
    def __init__(self, model, tokenizer, manager, max_context_len: int = 4096, device: str = "cuda"):
        super().__init__(model, tokenizer, manager, device=device)
        self.max_context_len = max_context_len
        self.max_blocks = (max_context_len + manager.block_size - 1) // manager.block_size

        # Static buffers: allocated once, contents overwritten every
        # call, never reallocated. This is what makes graph capture
        # valid across different sequences and different context lengths.
        self._static_cur_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._static_position_id = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._static_block_table = torch.zeros((1, self.max_blocks), dtype=torch.int32, device=device)
        self._static_context_len = torch.zeros((1,), dtype=torch.int32, device=device)
        self._static_write_page_id = torch.zeros((1,), dtype=torch.long, device=device)
        self._static_write_offset = torch.zeros((1,), dtype=torch.long, device=device)

        self._graph = None
        self._static_next_token = None

    def _forward_body(self) -> torch.Tensor:
        """The exact op sequence that gets captured. Reads ONLY from the
        static input buffers, never from a Python-side dynamic value --
        anything dynamic (which page to write to, how many valid blocks
        exist) must already be copied into a static buffer before this
        runs, whether this call is the capture or a replay."""
        hidden_states = self.embed_tokens(self._static_cur_token)
        cos, sin = self.rotary_emb(hidden_states, self._static_position_id)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            q, k, v = self._project_qkv(layer.self_attn, normed)
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

            k_for_cache = k[0, :, 0, :].unsqueeze(0)
            v_for_cache = v[0, :, 0, :].unsqueeze(0)
            # Vectorized, static-index write: the indices come from
            # static buffers filled just before this call, not built
            # fresh here -- graph-safe, unlike PagedEngine's version
            # which calls PagedKVCacheManager.write() (constructs fresh
            # index tensors each call).
            self.manager.k_cache[layer_idx][self._static_write_page_id, self._static_write_offset] = k_for_cache
            self.manager.v_cache[layer_idx][self._static_write_page_id, self._static_write_offset] = v_for_cache

            q_squeezed = q[:, :, 0, :]
            attn_out = paged_attention_decode(
                q_squeezed,
                self.manager.k_cache[layer_idx],
                self.manager.v_cache[layer_idx],
                self._static_block_table,
                self._static_context_len,
                self.manager.block_size,
                sm_scale=self.sm_scale,
            )
            attn_out = attn_out.reshape(1, 1, -1)
            attn_out = layer.self_attn.o_proj(attn_out)
            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(normed)
            hidden_states = residual + mlp_out

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _fill_static_buffers(self, cur_token: torch.Tensor, seq_id: int) -> None:
        """Dynamic, Python-side bookkeeping -- deliberately OUTSIDE the
        captured graph. Page allocation (reserve()) has real control
        flow (allocate a new page or not) that cannot be captured, but
        is cheap CPU-only work (per the Stage 4b lesson: the expensive
        case was GPU writes scaling with sequence length, not this)."""
        position_id = self.manager.context_len(seq_id)
        positions = self.manager.reserve(seq_id, 1)
        page_id, offset = positions[0]
        context_len_now = self.manager.context_len(seq_id)

        block_table_now = self.manager.block_table_tensor(seq_id, self.max_blocks)

        self._static_cur_token.copy_(cur_token)
        self._static_position_id.fill_(position_id)
        self._static_block_table.copy_(block_table_now.unsqueeze(0))
        self._static_context_len.fill_(context_len_now)
        self._static_write_page_id.fill_(page_id)
        self._static_write_offset.fill_(offset)

    def decode_step_graphed(self, cur_token: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Drop-in replacement for PagedEngine.decode_step, backed by a
        captured CUDA graph after the first call.

        No separate side-stream warmup before capture (an earlier
        version had one; removing it did not fix the bug described
        below, ruling it out as the cause).

        The real bug, found by elimination: the very first captured/
        replayed token consistently diverged from PagedEngine's real
        output, even though the exact same static-buffer logic was
        proven correct in plain eager mode. Swapping the Triton
        attention kernel for an equivalent pure-PyTorch computation
        inside the captured region produced the exact same wrong token
        -- ruling out Triton/CUDA-graph incompatibility as the cause,
        despite that being the more commonly-cited risk. The actual
        cause: CUDA graph capture's own execution pass does not
        guarantee a valid computed result -- it primarily records the
        kernel-launch structure; a correct result is only guaranteed
        after an actual replay. The first version of this code trusted
        the capture call's own output directly. Fixed by replaying once
        immediately after capture, before returning that first logical
        decode step's result too."""
        with torch.inference_mode():
            self._fill_static_buffers(cur_token, seq_id)

            if self._graph is None:
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_next_token = self._forward_body()
                # The capture pass's own output is not guaranteed valid --
                # CUDA graph capture primarily records the kernel-launch
                # structure; a correct computed result is only guaranteed
                # after an actual replay. Replay once immediately so this
                # first logical decode step also returns a real result,
                # not whatever the capture pass happened to leave behind.
                self._graph.replay()
            else:
                self._graph.replay()

            return self._static_next_token.clone()

    def generate_graphed(self, input_ids: torch.Tensor, max_new_tokens: int, seq_id: int = 0) -> torch.Tensor:
        next_token = self.prefill(input_ids, seq_id)
        tokens = [next_token]
        cur = next_token
        for _ in range(max_new_tokens - 1):
            cur = self.decode_step_graphed(cur, seq_id)
            tokens.append(cur)
        return torch.cat([input_ids] + tokens, dim=-1)
