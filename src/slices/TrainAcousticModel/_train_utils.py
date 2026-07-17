# src/slices/TrainAcousticModel/_train_utils.py — shared training helpers (LR schedule, seeding,
# ETA formatting, activation checkpointing) used by acoustic-model trainers (transducer path).
import math
import random

import torch


def _lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)  # cosine decay to 0
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


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
