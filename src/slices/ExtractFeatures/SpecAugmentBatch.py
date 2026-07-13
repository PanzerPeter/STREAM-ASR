# src/slices/ExtractFeatures/SpecAugmentBatch.py
# SpecAugment as a per-batch GPU op. Mirrors the old per-utterance "LD" policy (2 freq masks,
# adaptive time masking scaled by length) but runs on the collated batch already on-device, off the
# dataloader worker. Time masks stay within each sample's valid length so padding is never touched.
import torch

from src.shared_kernel.Config_Adapter import get_config


def apply_spec_augment_batch(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    cfg = get_config().augment
    out = features.clone()
    _, _, n_mels = out.shape
    for i in range(out.shape[0]):
        length = int(lengths[i])
        for _ in range(cfg.specaug_num_freq_masks):
            width = int(torch.randint(0, cfg.specaug_freq_width + 1, (1,)).item())
            if width == 0:
                continue
            start = int(torch.randint(0, max(1, n_mels - width), (1,)).item())
            out[i, :length, start : start + width] = 0.0
        num_time = min(cfg.specaug_max_time_masks, int(cfg.specaug_time_ratio * length))
        max_span = max(1, int(cfg.specaug_time_ratio * length))
        for _ in range(num_time):
            span = int(torch.randint(0, max_span + 1, (1,)).item())
            if span == 0:
                continue
            start = int(torch.randint(0, max(1, length - span), (1,)).item())
            out[i, start : start + span, :] = 0.0
    return out
