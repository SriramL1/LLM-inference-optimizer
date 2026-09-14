"""
Stage 4b: PagedEngine -- a manual, per-layer forward pass driven by our
own PagedKVCacheManager and kernels, instead of HF's built-in Cache
abstraction (option 2 from the Stage 4b scoping discussion: lower risk
than implementing a full transformers.Cache subclass, at the cost of not
getting generate()/beam-search compatibility for free).

Correctness-risk strategy: reimplementing a transformer's forward pass
by hand is exactly the kind of task where a subtle, silent bug (wrong
RoPE convention, wrong position id) produces plausible-looking garbage
rather than a crash. So this engine reuses HF's own tested pieces
wherever the risk of a from-scratch reimplementation would be high:

- q_proj/k_proj/v_proj/o_proj: the real nn.Linear layers, used directly.
- Rotary embeddings: the model's own rotary_emb module computes cos/sin;
  apply_rotary_pos_emb is imported by introspecting the actual installed
  self_attn module's own __module__ (not a hardcoded import path) --
  the same "read the real source, don't assume the API" lesson Stage 2's
  AttentionInterface naming collision taught.
- input_layernorm / post_attention_layernorm / mlp: called generically.
  This means it composes automatically with Stage 3's fused kernels if
  patch_model_with_fused_kernels() was already applied -- this engine
  doesn't care whether it's calling the original Qwen2RMSNorm/Qwen2MLP
  or Stage 3's FusedRMSNorm/FusedMLP, since it just calls them as plain
  callables either way.

Only the attention *computation* and KV-cache *storage* are genuinely
replaced -- prefill uses Stage 2's flash_attention kernel (self-attention
over the whole prompt), decode uses Stage 4's paged_attention_decode
kernel against the paged cache. That's the actual point of this stage.
"""
import importlib
from typing import Optional

import torch

from src.kernels.flash_attention_triton import flash_attention
from src.kernels.paged_attention_triton import paged_attention_decode
from src.kv_cache.block_manager import PagedKVCacheManager


def _resolve_apply_rotary_pos_emb(self_attn_module: torch.nn.Module):
    """Finds apply_rotary_pos_emb in the same module the real self_attn
    class was defined in, rather than assuming an import path -- import
    paths for these helper functions have moved around across
    transformers versions and model families."""
    module_path = type(self_attn_module).__module__
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "apply_rotary_pos_emb"):
        raise RuntimeError(
            f"Could not find apply_rotary_pos_emb in {module_path}. "
            "This model's module layout differs from what PagedEngine "
            "expects -- inspect the actual self_attn source (see the "
            "Stage 2 debugging notes for the pattern: read the real "
            "source before guessing at an API)."
        )
    return mod.apply_rotary_pos_emb


class PagedEngine:
    """Manual prefill/decode loop backed by a PagedKVCacheManager.

    One PagedEngine instance can drive multiple concurrent sequences
    (each with its own seq_id in the manager), though this Stage 4b pass
    only exercises it with single-sequence generation -- true concurrent
    variable-length serving is Stage 5's job (continuous batching).
    """

    def __init__(self, model: torch.nn.Module, tokenizer, manager: PagedKVCacheManager, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.device = device

        base_model = model.model  # Qwen2Model (the transformer stack, without the LM head)
        self.embed_tokens = base_model.embed_tokens
        self.layers = list(base_model.layers)
        self.final_norm = base_model.norm
        self.lm_head = model.lm_head

        if not hasattr(base_model, "rotary_emb"):
            raise RuntimeError(
                "Expected model.model.rotary_emb (shared rotary embedding module) -- "
                "not found. This model's architecture differs from what PagedEngine assumes."
            )
        self.rotary_emb = base_model.rotary_emb

        first_attn = self.layers[0].self_attn
        self.apply_rotary_pos_emb = _resolve_apply_rotary_pos_emb(first_attn)

        config = model.config
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_q_heads)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_layers = len(self.layers)
        self.sm_scale = getattr(first_attn, "scaling", self.head_dim ** -0.5)

    def _project_qkv(self, self_attn, hidden_states: torch.Tensor):
        """hidden_states: (batch, seq, hidden). Returns q, k, v each
        shaped (batch, num_*_heads, seq, head_dim)."""
        batch, seq_len, _ = hidden_states.shape
        q = self_attn.q_proj(hidden_states).view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self_attn.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self_attn.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, seq_id: int):
        """input_ids: (1, prompt_len) -- single sequence. Allocates
        seq_id in the manager, writes every layer's K/V for the full
        prompt, returns the greedy next token."""
        batch, prompt_len = input_ids.shape
        assert batch == 1, "PagedEngine currently drives one sequence at a time (Stage 5 adds real batching)"

        self.manager.allocate_sequence(seq_id)
        positions = self.manager.reserve(seq_id, prompt_len)

        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            q, k, v = self._project_qkv(layer.self_attn, normed)
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

            # (batch, num_kv_heads, seq, head_dim) -> (seq, num_kv_heads, head_dim) for the manager
            k_for_cache = k[0].transpose(0, 1).contiguous()
            v_for_cache = v[0].transpose(0, 1).contiguous()
            self.manager.write(seq_id, layer_idx, k_for_cache, v_for_cache, positions)

            attn_out = flash_attention(q, k, v, causal=True, sm_scale=self.sm_scale)
            attn_out = attn_out.transpose(1, 2).reshape(batch, prompt_len, -1)
            attn_out = layer.self_attn.o_proj(attn_out)

            hidden_states = residual + attn_out

            residual = hidden_states
            normed = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(normed)
            hidden_states = residual + mlp_out

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token

    @torch.inference_mode()
    def decode_step(self, cur_token: torch.Tensor, seq_id: int):
        """cur_token: (1, 1). Writes this step's K/V into the paged
        cache and runs the paged attention decode kernel for every
        layer. Returns the greedy next token."""
        batch = cur_token.shape[0]
        assert batch == 1

        position_id = self.manager.context_len(seq_id)  # this new token's absolute position
        positions = self.manager.reserve(seq_id, 1)

        hidden_states = self.embed_tokens(cur_token)
        position_ids = torch.tensor([[position_id]], device=self.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)

            q, k, v = self._project_qkv(layer.self_attn, normed)  # each (1, heads, 1, head_dim)
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

            k_for_cache = k[0].transpose(0, 1).contiguous()  # (1, num_kv_heads, head_dim)
            v_for_cache = v[0].transpose(0, 1).contiguous()
            self.manager.write(seq_id, layer_idx, k_for_cache, v_for_cache, positions)

            context_len_now = self.manager.context_len(seq_id)  # includes the token just written
            max_blocks = (context_len_now + self.manager.block_size - 1) // self.manager.block_size
            block_table = self.manager.block_table_tensor(seq_id, max_blocks).unsqueeze(0)
            context_lens_tensor = torch.tensor([context_len_now], dtype=torch.int32, device=self.device)

            q_squeezed = q[:, :, 0, :]  # (1, num_q_heads, head_dim) -- single decode query
            attn_out = paged_attention_decode(
                q_squeezed,
                self.manager.k_cache[layer_idx],
                self.manager.v_cache[layer_idx],
                block_table,
                context_lens_tensor,
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
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, seq_id: int = 0):
        """Convenience wrapper: greedy-decode max_new_tokens tokens.
        Returns the full token sequence (prompt + generated)."""
        next_token = self.prefill(input_ids, seq_id)
        tokens = [next_token]
        cur = next_token
        for _ in range(max_new_tokens - 1):
            cur = self.decode_step(cur, seq_id)
            tokens.append(cur)
        return torch.cat([input_ids] + tokens, dim=-1)
