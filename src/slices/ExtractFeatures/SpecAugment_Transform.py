# src/slices/ExtractFeatures/SpecAugment_Transform.py
import torch

from src.shared_kernel.Config_Adapter import get_config


def apply_spec_augment(mel: torch.Tensor) -> torch.Tensor:
    cfg = get_config().augment
    out = mel.clone()
    num_frames, num_mels = out.shape

    for _ in range(cfg.specaug_num_freq_masks):
        width = int(torch.randint(0, cfg.specaug_freq_width + 1, (1,)).item())
        if width == 0:
            continue
        start = int(torch.randint(0, max(1, num_mels - width), (1,)).item())
        out[:, start : start + width] = 0.0

    # Adaptive time masking: mask count scales with utterance length (SpecAugment "LD" policy).
    num_time_masks = min(cfg.specaug_max_time_masks, int(cfg.specaug_time_ratio * num_frames))
    max_span = max(1, int(cfg.specaug_time_ratio * num_frames))
    for _ in range(num_time_masks):
        span = int(torch.randint(0, max_span + 1, (1,)).item())
        if span == 0:
            continue
        start = int(torch.randint(0, max(1, num_frames - span), (1,)).item())
        out[start : start + span, :] = 0.0

    return out
