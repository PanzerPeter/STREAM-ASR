# Maximal Update Parametrization helpers (Yang & Hu, muTransfer). muP keeps per-layer activation and
# update scale width-invariant so a learning rate tuned at a small proxy width transfers to a larger
# target width — the mechanism for a future cheap encoder-size search (SP3). Hidden Adam LRs scale
# 1/width via the per-param _mup_lr_scale tag; the readout uses a 1/fan_in forward multiplier and
# zero-init. All tags default to no-op so a non-muP model is unaffected.
import math

import torch.nn as nn


def mup_linear_(linear: nn.Linear, base_fan_in: int) -> None:
    fan_in = linear.weight.shape[1]
    std = 1.0 / math.sqrt(fan_in)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    scale = base_fan_in / fan_in
    linear.weight._mup_lr_scale = scale  # type: ignore[attr-defined]
    if linear.bias is not None:
        linear.bias._mup_lr_scale = 1.0  # type: ignore[attr-defined]


def mup_readout_(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)  # canonical muP zero-init readout
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    linear._mup_mult = 1.0 / linear.weight.shape[1]  # type: ignore[assignment]
    linear.weight._mup_lr_scale = 1.0  # type: ignore[attr-defined]
    if linear.bias is not None:
        linear.bias._mup_lr_scale = 1.0  # type: ignore[attr-defined]


def mup_lr_scale(param: nn.Parameter) -> float:
    return float(getattr(param, "_mup_lr_scale", 1.0))
