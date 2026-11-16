# pyright: reportPrivateImportUsage=false
#   torch's `_foreach_*` multi-tensor ops are documented public API with private-looking
#   names -- torch's own optimizers call them -- but they are not in `torch.__all__`, so a
#   strict checker flags every use. mypy accepts them; only this rule needs the exemption.
# Gradient clipping for a model whose gates/vectors and weight matrices need different treatment,
# plus the parameter partition that separates them. Pure tensor ops on a parameter list.
import torch


def guarded_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The parameters whose gradient norm tracks model health: the weight matrices (ndim >= 2).

    A vector or scalar parameter (a bias, a gate pre-activation, a QK gain) multiplies a whole
    [B, T, C] activation, so its gradient is a reduction over millions of terms, sized by the batch
    rather than by the model. One such gate has been measured carrying ~95 % of the global norm and
    driving false "gradient runaway" aborts while the matrices' own gradient scale was flat.
    """
    return [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]


def unguarded_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The complement of `guarded_parameters`, so the two sets can be clipped SEPARATELY.

    One `clip_grad_norm_` over everything rescales every gradient by `grad_clip / global_norm`,
    and that norm is this set's: one gate then decides the factor every weight matrix is multiplied
    by, swinging ~100x step to step. Muon renormalises within a step, but its momentum is an EMA
    across steps, so a per-step-varying factor reweights it (MEASURED: Muon momentum norms fell
    1.30 -> 0.19 while the matrices' raw gradient norm never moved).
    """
    return [p for p in model.parameters() if p.requires_grad and p.ndim < 2]


def clip_grads_per_tensor(params: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip each parameter's gradient to `max_norm` ON ITS OWN. Returns the group's pre-clip norm.

    A shared-norm clip inside this group only relocates the coupling: one gate can sit at exactly
    `grad_clip` every step and rescale every other vector with it. Adam is invariant to
    a constant rescale but not to one that swings step to step, because its moments are EMAs across
    steps. Three `_foreach` launches for the whole list.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    norms = torch.stack(torch._foreach_norm(grads))
    # clamp(max=1) so a gradient already inside the bound is multiplied by exactly 1.0, not by
    # max_norm/norm; the eps keeps a zero-gradient tensor from producing inf * 0.
    scales = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, list(scales.unbind()))
    return torch.linalg.vector_norm(norms)


def grad_norm_of(params: list[torch.nn.Parameter]) -> torch.Tensor:
    """L2 norm over those params' gradients, one fused multi-tensor reduction, left on device so the
    caller batches the device->host sync into the one it already does at `log_every`."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
