# src/slices/PretrainEncoder/MelOnlyCollator.py
# Zero-pads a batch of ragged [T, 80] log-mel tensors to [B, Tmax, 80] — BEST-RQ masks/quantizes
# on the padded grid, so lengths travel alongside to exclude pad frames from the pretrain loss.
import torch


def collate_mels(samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([m.shape[0] for m in samples], dtype=torch.long)
    t_max = int(lengths.max())
    n_mels = samples[0].shape[1]
    feats = torch.zeros(len(samples), t_max, n_mels, dtype=torch.float32)
    for i, m in enumerate(samples):
        feats[i, : m.shape[0]] = m
    return feats, lengths
