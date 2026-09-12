"""
Stage 3: wire the fused RMSNorm and SwiGLU kernels into the real model.

Design note -- this is the second design, not the first. The first
version monkey-patched .forward directly on existing module instances
(via types.MethodType), the same technique commonly used and generally
safe for nn.Module. It broke the model into producing NaN -- and,
decisively, a *pure passthrough* replacement (one that called the
original class method and computed nothing different at all) broke it
identically. That proved the bug had nothing to do with kernel math or
even the replacement's behavior: reassigning .forward on an existing
instance was itself the problem, most likely an interaction with some
graph-capture/compilation mechanism in this transformers/PyTorch version
that dispatches forward calls in a way that doesn't respect a plain
instance-level attribute override.

The fix: don't monkey-patch existing instances at all. Replace the whole
child module in its parent's module registry with a new, properly
constructed nn.Module subclass instead -- the standard technique used by
quantization/PEFT-style libraries for exactly this kind of model surgery,
and structurally different enough to avoid whatever the instance-patching
approach was colliding with.
"""
import torch

from src.kernels.fused_rmsnorm_triton import fused_rmsnorm
from src.kernels.fused_swiglu_triton import fused_silu_mul

RMSNORM_CLASS_NAMES = {"Qwen2RMSNorm", "LlamaRMSNorm", "MistralRMSNorm", "RMSNorm"}
MLP_CLASS_NAMES = {"Qwen2MLP", "LlamaMLP", "MistralMLP"}


class FusedRMSNorm(torch.nn.Module):
    """Wraps an existing RMSNorm module's weight/eps, replacing its
    computation with the fused Triton kernel. Reuses the original
    nn.Parameter object directly (not a copy) so the real trained
    weights are preserved exactly."""

    def __init__(self, orig_module: torch.nn.Module):
        super().__init__()
        self.weight = orig_module.weight
        self.variance_epsilon = orig_module.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(hidden_states, self.weight, self.variance_epsilon)


class FusedMLP(torch.nn.Module):
    """Wraps an existing MLP module's three projections, replacing the
    silu(gate) * up step with the fused kernel. The projections
    themselves (nn.Linear, with their real trained weights) are reused
    directly, not recreated."""

    def __init__(self, orig_module: torch.nn.Module):
        super().__init__()
        self.gate_proj = orig_module.gate_proj
        self.up_proj = orig_module.up_proj
        self.down_proj = orig_module.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = fused_silu_mul(gate, up)
        return self.down_proj(activated)


def patch_model_with_fused_kernels(
    model: torch.nn.Module,
    patch_norm: bool = True,
    patch_mlp: bool = True,
) -> dict:
    """Walks the model, replacing matching RMSNorm and/or MLP child
    modules in-place (via setattr on their parent, which correctly
    updates the parent's module registry) with fused equivalents.

    Returns a dict with counts of each type replaced -- callers should
    check these are both > 0 (for architectures this covers) rather than
    assume success, since an unrecognized architecture silently replaces
    nothing.

    patch_norm / patch_mlp let each kernel be enabled independently --
    useful for bisecting a bug to one specifically."""
    counts = {"rmsnorm": 0, "mlp": 0}

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            cls_name = child.__class__.__name__

            if patch_norm and cls_name in RMSNORM_CLASS_NAMES and hasattr(child, "weight") and hasattr(child, "variance_epsilon"):
                setattr(parent, child_name, FusedRMSNorm(child))
                counts["rmsnorm"] += 1

            elif patch_mlp and cls_name in MLP_CLASS_NAMES and all(
                hasattr(child, attr) for attr in ("gate_proj", "up_proj", "down_proj")
            ):
                setattr(parent, child_name, FusedMLP(child))
                counts["mlp"] += 1

    return counts
