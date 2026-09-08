"""
Baseline inference engine: a minimal, unoptimized autoregressive loop.

This is intentionally the "slow but obviously correct" reference that every
later stage (fused attention, paged KV-cache, continuous batching, CUDA
graphs, quantization, speculative decoding) gets benchmarked against. It
uses HuggingFace's built-in KV-cache and eager attention -- no custom
kernels yet. Prefill and decode are exposed as separate methods on purpose:
those two phases have very different performance characteristics (compute-
bound vs memory-bandwidth-bound), and the profiler needs to time them
independently.
"""
import torch


class BaselineEngine:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor):
        """Process the full prompt in one forward pass.

        Returns the KV-cache and the first generated token (greedy).
        """
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return past_key_values, next_token

    @torch.inference_mode()
    def decode_step(self, cur_token: torch.Tensor, past_key_values):
        """Process a single new token against the existing KV-cache."""
        outputs = self.model(
            input_ids=cur_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return outputs.past_key_values, next_token

    def generate_text(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Convenience wrapper for eyeballing output. Not used in benchmarks
        (the profiler drives prefill/decode directly so it can time each
        step with CUDA events)."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        past_key_values, next_token = self.prefill(input_ids)
        tokens = [next_token]
        cur = next_token
        for _ in range(max_new_tokens - 1):
            past_key_values, cur = self.decode_step(cur, past_key_values)
            tokens.append(cur)
        all_tokens = torch.cat([input_ids] + tokens, dim=-1)
        return self.tokenizer.decode(all_tokens[0], skip_special_tokens=True)
