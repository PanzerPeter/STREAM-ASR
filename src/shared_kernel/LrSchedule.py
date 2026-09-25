# Peak-LR shape multiplier for the acoustic trainers: linear warmup into a warmup-stable-decay
# trapezoid. Kept as a pure function of the step so every trainer applies the same schedule to its
# own per-group peaks.
import math


def lr_at(
    step: int,
    peak: float,
    warmup: int,
    total: int,
    decay_frac: float = 0.25,
    min_ratio: float = 0.0,
) -> float:
    """Linear warmup, then hold `peak` until the last `decay_frac` of the budget, then a 1-sqrt
    anneal to `peak * min_ratio` exactly at `total` (Hägele et al. 2024: the strongest of the WSD
    decay shapes).

    `min_ratio > 0` keeps the tail steps doing work and keeps the rolling snapshots that
    `scripts/average_checkpoints.py` means over meaningfully different from one another. Raising
    `total` mid-run only moves the decay window. Under the cosine schedule this replaced, `total`
    was the curve, so bumping it re-heated the LR at the resume step -- which is how the
    120k -> 175k bump that produced the v1.0 checkpoint accidentally became a two-cycle schedule.
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    decay_steps = max(1, round(decay_frac * total))
    decay_start = max(warmup, total - decay_steps)
    if step < decay_start:
        return peak
    frac = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(frac)))
