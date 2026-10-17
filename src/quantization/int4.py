"""
Stage 7b: INT4 group-wise weight quantization of a loaded model.

Same integration approach as INT8 (module replacement via setattr on the
parent), same scope: every nn.Linear inside the transformer layers;
embeddings and the tied lm_head stay fp16.

Plain round-to-nearest (RTN) quantization with max-abs group scales. No
calibration data, no error compensation (GPTQ) and no activation-aware
scaling (AWQ) -- those are what production INT4 uses to recover accuracy,
and are explicitly out of scope. Expect visibly more accuracy loss than
INT8; the tests bound it and the docs report the measured numbers.
"""
import gc
from typing import Optional

import torch
import torch.nn as nn

from src.kernels.w4a16_matmul_triton import (
    GROUP_SIZE, pack_int4, w4a16_linear, dequantize_int4,
)


class QuantizedLinearInt4(nn.Module):
    """Drop-in nn.Linear replacement: packed 4-bit weights, fp16 group
    scales of shape (K//128, N), optional bias. Inference-only buffers."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.register_buffer("weight_packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_features = packed.shape[0]
        self.in_features = packed.shape[1] * 2

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearInt4":
        w = linear.weight.detach().float()                 # (N, K)
        N, K = w.shape
        assert K % GROUP_SIZE == 0, f"in_features {K} must be a multiple of {GROUP_SIZE}"
        wg = w.reshape(N, K // GROUP_SIZE, GROUP_SIZE)
        scale = (wg.abs().amax(dim=2) / 7.0).clamp(min=1e-5)          # (N, G)
        scale16 = scale.half()                                          # what is actually stored
        q = torch.round(wg / scale16.float()[:, :, None]).clamp(-7, 7) + 8
        packed = pack_int4(q.reshape(N, K).to(torch.uint8))
        scales = scale16.t().contiguous()                               # (G, N)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(packed, scales, bias)

    def dequantized_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_int4(self.weight_packed, self.scales).to(dtype)

    def num_bytes(self) -> int:
        total = self.weight_packed.numel() * self.weight_packed.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w4a16_linear(x, self.weight_packed, self.scales, self.bias)


def _linear_bytes(linear: nn.Linear) -> int:
    total = linear.weight.numel() * linear.weight.element_size()
    if linear.bias is not None:
        total += linear.bias.numel() * linear.bias.element_size()
    return total


def quantize_model_int4(model: nn.Module) -> dict:
    """Replaces every nn.Linear inside model.model.layers with a
    QuantizedLinearInt4, in place. Returns stats; callers should check
    layers_quantized > 0 rather than assume it worked."""
    layers_quantized = 0
    bytes_before = 0
    bytes_after = 0

    for parent in list(model.model.layers.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                bytes_before += _linear_bytes(child)
                quantized = QuantizedLinearInt4.from_linear(child)
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
