# Projected gradient descent for the model's constrained parameters.
#
# Two parameters in this model carry a hard constraint -- `ZipformerStack.bypass` in [0, 1] and
# `BiasNorm.log_scale` in its configured window -- and both enforce it with `clamp` inside
# `forward`. That alone is not enough, because `clamp`'s gradient is exactly 0 outside its bounds:
# a parameter the optimizer pushes past one receives no gradient ever again and can never come
# back. Re-projecting after each step leaves it resting exactly ON the bound, where gradient still
# flows, which is ordinary projected gradient descent.
#
# Deliberately NOT done inside `forward`: an in-place write to a leaf between two forwards that
# share one backward invalidates the first forward's saved tensor, which is exactly what the CR-CTC
# two-view path does, and it raises "modified by an inplace operation".
import torch.nn as nn


def project_constraints(module: nn.Module) -> None:
    """Call `project()` on every submodule that has one. Trainers call this after `optimizer.step`.

    Duck-typed rather than an isinstance dispatch so it needs no import of any slice's modules --
    it lives in the shared kernel and both the BEST-RQ and transducer trainers call it on their
    whole model, which is what reaches the predictor's `BiasNorm` as well as the encoder's 97.
    Walking `modules()` also finds the stacks when the trainer has wrapped each one in
    `_Checkpointed` for activation checkpointing.
    """
    for child in module.modules():
        project = getattr(child, "project", None)
        if project is not None:
            project()
