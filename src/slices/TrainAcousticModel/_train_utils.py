# Shared training helpers (LR schedule, seeding, ETA formatting, activation checkpointing) used by
# the acoustic-model trainers.
import math
import random

import torch


def _lr_at(
    step: int,
    peak: float,
    warmup: int,
    total: int,
    schedule: str = "cosine",
    stable_ratio: float = 1.0,
    decay_frac: float = 0.25,
    min_ratio: float = 0.0,
) -> float:
    """Linear warmup, then either a cosine anneal or a warmup-stable-decay (WSD) trapezoid.

    `total` is the anneal anchor as well as the step budget, so both shapes land on
    `peak * min_ratio` exactly at `total`. `min_ratio > 0` keeps the tail steps doing work
    (a cosine that reaches 0 spends its last few thousand steps not training) and keeps the
    rolling snapshots that `scripts/average_checkpoints.py` means over meaningfully different
    from one another.

    WSD holds `peak * stable_ratio` from the end of warmup until the last `decay_frac` of the
    budget, then anneals over that window with a 1-sqrt profile (Hägele et al. 2024: the
    strongest of the WSD decay shapes). Two properties cosine does not have:
      * the LR stays high through the middle of the run instead of being at half peak by the
        halfway point, and the anneal is short and sharp;
      * raising `total_steps` mid-run only moves the decay window. Under cosine, `total` is the
        curve, so bumping it re-heats the LR at the resume step -- which is how the 120k -> 175k
        bump that produced the shipped checkpoint accidentally became a two-cycle schedule.
    `stable_ratio < 1.0` exists for the same reason in reverse: when switching an in-flight
    cosine run over to WSD, set it to the LR fraction that run had already decayed to so the
    restart does not re-heat the encoder.
    """
    if step < warmup:
        return peak * stable_ratio * step / max(1, warmup)
    if schedule == "wsd":
        decay_steps = max(1, round(decay_frac * total))
        decay_start = max(warmup, total - decay_steps)
        if step < decay_start:
            return peak * stable_ratio
        frac = min(1.0, (step - decay_start) / max(1, total - decay_start))
        return peak * (min_ratio + (stable_ratio - min_ratio) * (1.0 - math.sqrt(frac)))
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * stable_ratio * (min_ratio + (1.0 - min_ratio) * cosine)


def _seed_all(seed: int) -> None:
    # Seed model init, worker augmentation, and batch order so the blank-collapse escape (an init-
    # sensitive knife-edge) is reproducible. torch.manual_seed also fixes the DataLoader workers'
    # per-worker seeds (PyTorch derives them from the main generator), so SpecAugment/speed-perturb
    # become deterministic too. use_deterministic_algorithms is deliberately NOT set: cuDNN's CTC
    # has no deterministic kernel and would raise.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class _Checkpointed(torch.nn.Module):
    """Wraps a stack so its forward runs under activation checkpointing."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, lengths, pad_mask, chunk_size=0):
        return torch.utils.checkpoint.checkpoint(
            self.module, x, lengths, pad_mask, chunk_size, use_reentrant=False
        )
