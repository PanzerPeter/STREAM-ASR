# pyright: reportPrivateImportUsage=false
#   torch's `_foreach_*` multi-tensor ops are documented public API with private-looking
#   names -- torch's own optimizers call them -- but they are not in `torch.__all__`, so a
#   strict checker flags every use. mypy accepts them; only this rule needs the exemption.
# Gradient clipping for a model whose scalar gates and weight matrices need different treatment,
# plus the parameter partition that separates them. Pure tensor ops on a parameter list, so both
# acoustic trainers (BEST-RQ pretrain and the transducer) reach the same implementation.
import torch


def guarded_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The parameters whose gradient norm actually tracks model health: the weight matrices.

    Everything with `ndim < 2` -- biases, `BiasNorm.log_scale`, and above all the six scalar
    `ZipformerStack.bypass` gates -- is a per-element parameter multiplying a whole `[B, T, C]`
    activation, so its gradient is a reduction over millions of terms. Its magnitude is set by the
    batch's shape, not by anything about the model, and it dominates the global norm by ~30x.

    MEASURED from the optimizer moments in the checkpoints of the two runs aborted for
    "gradient runaway" (AdamW `sqrt(sum exp_avg_sq)` per tensor, so proportional to each tensor's
    recent gradient scale):

        checkpoint                     Muon (135 matrices)   AdamW total   stacks.3.bypass
        step394200 (healthy resume)          0.7315            2.429           2.063
        step410400 (aborted "runaway")       0.7535            3.287           3.023
        400k_predivergence (median 11.0)     0.7241            2.981           2.766
        step415800 (deep in the old abort)   0.1743            4.994           4.987

    The encoder's own gradient scale is FLAT across every one of those, including both moments the
    guard called a runaway; ~95 % of `train/grad_norm` is one scalar. Both aborts were false
    positives -- the second fired while dev ctc-WER was setting the run's record (0.0733 -> 0.0680)
    and the loss was falling. Filtering to `ndim >= 2` is what makes the guard measure the quantity
    its docstring describes.
    """
    return [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]


def unguarded_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The complement of `guarded_parameters`: biases, norm gains and the scalar gates.

    Exists so the trainer can clip the two sets SEPARATELY. A single
    `clip_grad_norm_(model.parameters(), ...)` computes one global norm and rescales every gradient
    by `grad_clip / norm` -- and that norm is ~99.9 % this set (a per-element parameter multiplying
    a whole `[B, T, C]` activation has a gradient sized by the batch). So one scalar gate decides
    the factor every weight matrix's gradient is multiplied by, and that factor swings ~100x from
    step to step.

    Muon's Newton-Schulz renormalises within a step, so a *constant* rescale would be invisible to
    it. A per-step-varying one is not: `momentum_buffer` is an EMA across steps, so rescaling each
    step's contribution reweights it, and the encoder's update direction ends up chosen by whichever
    steps happened to have a small gate gradient. MEASURED on the 600k run between step 275,400 and
    297,000, while `grad_norm_guarded` held flat at ~0.9 and the global norm went 2.0 -> 334:
    Muon's momentum norms collapsed 1.30 -> 0.19 overall (`encoder.frontend.conv1.weight`
    0.98 -> 0.12). That collapse is the clip, not the encoder -- it is the same signature the
    2026-08-05 post-mortem recorded as "the encoder gradient collapsing 4x" and read as a cause.
    """
    return [p for p in model.parameters() if p.requires_grad and p.ndim < 2]


def clip_grads_per_tensor(params: list[torch.nn.Parameter], max_norm: float) -> torch.Tensor:
    """Clip each parameter's gradient to `max_norm` ON ITS OWN. Returns the group's pre-clip norm.

    `clip_grad_norm_` computes ONE norm over the whole list and rescales every gradient by
    `max_norm / norm`, which couples parameters that have nothing to do with each other. For the
    scalar gates that is not a subtlety, it is the whole behaviour: a per-element parameter
    multiplying a whole `[B, T, C]` activation has a gradient that is a reduction over millions of
    terms, so one of them owns the group norm outright.

    MEASURED off `transducer_last.pt` (step 311,326), AdamW `sqrt(sum exp_avg_sq)` per tensor, i.e.
    each tensor's recent post-clip gradient scale:

        encoder.stacks.3.bypass    4.9998      <- grad_clip is 5.0
        encoder.stacks.1.bypass    0.0248
        encoder.stacks.2.bypass    0.0092
        encoder.stacks.0.bypass    0.0066

    One gate is consuming the entire clip budget every step, so the other 414 scalars -- every bias
    and all 98 `BiasNorm.log_scale` -- are rescaled by a factor set by whatever that gate's gradient
    happened to be. Splitting the clip in two (matrices | scalars) did not fix that, it relocated
    it: the factor used to poison the Muon matrices and now poisons the AdamW scalars instead.

    A CONSTANT rescale would be invisible to Adam, which is scale-invariant per parameter. This one
    is not constant, it swings ~100x step to step, and `exp_avg`/`exp_avg_sq` are EMAs ACROSS steps
    -- so `m/sqrt(v)` ends up weighted toward whichever steps happened to have a small gate
    gradient. Clipping each tensor against its own norm keeps `grad_clip` meaning what it says and
    leaves the coupling out.

    Three `_foreach` launches for the whole list, so this costs the same as the fused clip it
    replaces. The returned norm is the pre-clip global one, which is what `train/grad_norm` has
    always logged.
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
    """L2 norm over those params' gradients, as one fused multi-tensor reduction, left on device.

    `_foreach_norm` folds ~200 per-parameter reductions into one launch, and the norm of the
    per-tensor norms is the global L2 norm. Kept as a device tensor so the caller can batch the
    device->host sync into the one it already does at `log_every`.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
