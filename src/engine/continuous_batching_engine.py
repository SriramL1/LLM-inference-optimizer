"""
Stage 5: continuous batching -- batch multiple concurrent sequences'
decode steps into a single forward pass per iteration.

Decode is memory-bandwidth-bound: each step re-reads the entire model's
weights to produce one token per sequence. Stage 1 already measured that
batching amortizes this cost near-linearly (47 -> 384 tok/s from batch 1
to 8). Continuous batching is what makes that possible in a real serving
system, where requests arrive and finish at different times rather than
all starting and ending together -- a scheduler admits new sequences
into the batch as soon as a slot frees up, instead of waiting for the
whole batch to finish (static batching).

ContinuousBatchingEngine extends PagedEngine rather than duplicating it:
prefill is unchanged (still one sequence at a time -- mixing prefill's
many new tokens with decode's one-token-per-sequence shape into a single
batched kernel call is "chunked prefill", a further real-systems
technique intentionally out of scope here). The new piece is
decode_step_batched, which reuses the SAME paged_attention_decode kernel
Stage 4 already validated for variable-length batches -- that's exactly
the shape this needs, so no new kernel work was required, only new
engine/scheduling logic around the existing one.
"""
from typing import Dict

import torch

from src.engine.paged_engine import PagedEngine
from src.kernels.paged_attention_triton import paged_attention_decode


class ContinuousBatchingEngine(PagedEngine):
    def decode_step_batched(self, seq_tokens: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """seq_tokens: {seq_id: cur_token (1,1)} for every sequence
        currently in the active decode pool (each sequence may be at a
        different context length -- that's the whole point). Returns
        {seq_id: next_token} for each."""
        seq_ids = list(seq_tokens.keys())
        batch = len(seq_ids)

        cur_tokens = torch.cat([seq_tokens[sid] for sid in seq_ids], dim=0)  # (batch, 1)
        position_ids = torch.tensor(
            [[self.manager.context_len(sid)] for sid in seq_ids], device=self.device
        )  # (batch, 1) -- each sequence's own current position, may differ across the batch

        # Cheap CPU-only bookkeeping (unlike Stage 4b's original per-token
        # write bug, this loop is O(batch), not O(sequence_length) -- the
        # expensive case was already fixed at the source).
        positions_per_seq = [self.manager.reserve(sid, 1) for sid in seq_ids]

        hidden_states = self.embed_tokens(cur_tokens)  # (batch, 1, hidden)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            q, k, v = self._project_qkv(layer.self_attn, normed)  # (batch, heads, 1, head_dim)
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

            for i, sid in enumerate(seq_ids):
                k_for_cache = k[i, :, 0, :].unsqueeze(0)  # (1, num_kv_heads, head_dim)
                v_for_cache = v[i, :, 0, :].unsqueeze(0)
                self.manager.write(sid, layer_idx, k_for_cache, v_for_cache, positions_per_seq[i])

            context_lens = [self.manager.context_len(sid) for sid in seq_ids]
            max_blocks = max((c + self.manager.block_size - 1) // self.manager.block_size for c in context_lens)
            block_table = self.manager.batch_block_table_tensor(seq_ids, max_blocks)
            context_lens_tensor = self.manager.batch_context_lens_tensor(seq_ids)

            q_squeezed = q[:, :, 0, :]  # (batch, num_q_heads, head_dim)
            attn_out = paged_attention_decode(
                q_squeezed,
                self.manager.k_cache[layer_idx],
                self.manager.v_cache[layer_idx],
                block_table,
                context_lens_tensor,
                self.manager.block_size,
                sm_scale=self.sm_scale,
            )
            attn_out = attn_out.reshape(batch, 1, -1)
            attn_out = layer.self_attn.o_proj(attn_out)

            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(normed)
            hidden_states = residual + mlp_out

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)

        return {sid: next_tokens[i : i + 1] for i, sid in enumerate(seq_ids)}
