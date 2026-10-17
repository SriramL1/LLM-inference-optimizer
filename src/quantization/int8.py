"""
Stage 7: INT8 weight quantization of a loaded model.

Integration approach follows the Stage 3 lesson: replace whole child
modules in their parent's registry (setattr on the parent) rather than
monkey-patching .forward on existing instances, which was proven fragile
in this environment.

Composition: PagedEngine / GraphedDecodeEngine / ContinuousBatchingEngine
call q_proj/k_proj/v_proj/o_proj and the MLP projections as plain
callables, looked up fresh from the module each call -- so a model
quantized with quantize_model_int8() works with all of them with zero
changes to those engines. The same holds for HuggingFace's own forward.

Scope, stated plainly:
- Quantizes every nn.Linear inside the transformer layers (q/k/v/o and
  gate/up/down projections) -- the bulk of the weights.
- Does NOT quantize embed_tokens or lm_head. For Qwen2.5-1.5B these are
  tied (one shared matrix), so quantizing lm_head would mean untying it
  from the embedding, and logits precision matters more than the ~1-2ms
  it would save. Documented limitation, not an oversight.
"""
import gc
from typing import Optional

import torch
import torch.nn as nn

from src.kernels.w8a16_matmul_triton import w8a16_linear


class QuantizedLinearInt8(nn.Module):
    """Drop-in replacement for nn.Linear holding int8 weights plus a
    per-output-channel float32 scale. Weights and scales are buffers, not
    Parameters: this is an inference-only module."""

    def __init__(self, weight_q: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)  # None is allowed
        self.out_features, self.in_features = weight_q.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearInt8":
        w = linear.weight.detach().float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        w_q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(w_q.contiguous(), scale.float().contiguous(), bias)

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.weight_q.float() * self.scale[:, None]).to(dtype)

    def num_bytes(self) -> int:
        total = self.weight_q.numel() * self.weight_q.element_size()
        total += self.scale.numel() * self.scale.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w8a16_linear(x, self.weight_q, self.scale, self.bias)


def _linear_bytes(linear: nn.Linear) -> int:
    total = linear.weight.numel() * linear.weight.element_size()
    if linear.bias is not None:
        total += linear.bias.numel() * linear.bias.element_size()
    return total


def quantize_model_int8(model: nn.Module) -> dict:
    """Replaces every nn.Linear inside model.model.layers with a
    QuantizedLinearInt8, in place. Returns a stats dict -- callers should
    check layers_quantized > 0 rather than assume it worked (same
    silent-no-op failure mode Stage 3's patch function documents)."""
    layers_root = model.model.layers
    layers_quantized = 0
    bytes_before = 0
    bytes_after = 0

    for parent in list(layers_root.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                bytes_before += _linear_bytes(child)
                quantized = QuantizedLinearInt8.from_linear(child)
                bytes_after += quantized.num_bytes()
                setattr(parent, child_name, quantized)
                layers_quantized += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "layers_quantized": layers_quantized,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }
