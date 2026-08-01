# SpecAugment as a per-batch GPU op. Mirrors the per-utterance "LD" policy (2 freq masks, adaptive
# time masking scaled by length) but runs on the collated batch already on-device, off the
# dataloader worker. Time masks stay within each sample's valid length so padding is never touched.
#
# Fully vectorised: the mask geometry is drawn as [B, K] tensors and reduced into one boolean
# [B, T, n_mels] mask, so the cost is a fixed handful of kernels regardless of batch size. Masking
# per utterance instead would cost an `int(lengths[i])` device->host sync each (twice per step
# under CR-CTC) plus a slice-assign launch per mask -- pipeline bubbles on the critical path. The
# sampled distribution is the standard one: widths and starts are uniform over the same integer
# ranges, drawn as floor(rand * n) rather than randint(0, n) to stay on-device.
import torch

from src.shared_kernel.Config_Adapter import get_config


def _uniform_int(high: torch.Tensor) -> torch.Tensor:
    # Uniform over {0, ..., high-1} elementwise, matching torch.randint(0, high). torch.rand is
    # half-open on [0, 1), so the floor can never reach `high`. Drawn in fp32 regardless of the
    # feature dtype: bf16 has 8 mantissa bits and would quantise the draw into ~256 buckets.
    return (torch.rand(high.shape, device=high.device, dtype=torch.float32) * high).long()


def apply_spec_augment_batch(features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    cfg = get_config().augment
    batch, num_frames, n_mels = features.shape
    device = features.device
    valid = lengths.to(device).view(batch, 1)  # [B, 1], kept on-device -- never .item()'d
    frames = torch.arange(num_frames, device=device).view(1, num_frames, 1)
    bins = torch.arange(n_mels, device=device).view(1, 1, n_mels)

    # Frequency masks: width ~ U{0..specaug_freq_width}, start ~ U{0..max(1, n_mels-width)-1}.
    freq_slots = torch.full(
        (batch, cfg.specaug_num_freq_masks), cfg.specaug_freq_width + 1, device=device
    )
    width = _uniform_int(freq_slots)
    start_f = _uniform_int((n_mels - width).clamp_(min=1))
    freq_masked = (
        (bins.unsqueeze(1) >= start_f.unsqueeze(2).unsqueeze(3))
        & (bins.unsqueeze(1) < (start_f + width).unsqueeze(2).unsqueeze(3))
    ).any(
        dim=1
    )  # [B, 1, n_mels]

    # Time masks: count and span both scale with the utterance's own valid length --
    # num_time = min(max_time_masks, int(ratio*L)), span ~ U{0..max(1, int(ratio*L))}.
    span_cap = (valid.float() * cfg.specaug_time_ratio).long()
    num_time = span_cap.clamp(max=cfg.specaug_max_time_masks)
    max_span = span_cap.clamp(min=1)
    span = _uniform_int((max_span + 1).expand(batch, cfg.specaug_max_time_masks).contiguous())
    # Every utterance draws max_time_masks slots but only num_time of them are real; zeroing the
    # span of the surplus slots is the vectorised equivalent of never drawing them.
    span = torch.where(
        torch.arange(cfg.specaug_max_time_masks, device=device).view(1, -1) < num_time,
        span,
        torch.zeros_like(span),
    )
    start_t = _uniform_int((valid - span).clamp_(min=1))
    time_masked = (
        (frames.unsqueeze(1) >= start_t.unsqueeze(2).unsqueeze(3))
        & (frames.unsqueeze(1) < (start_t + span).unsqueeze(2).unsqueeze(3))
    ).any(
        dim=1
    )  # [B, num_frames, 1]

    # Both mask families are confined to t < length. Time starts are already drawn so that
    # start+span <= length; this final AND is what keeps the freq masks off the padding too.
    in_valid = frames < valid.view(batch, 1, 1)
    return features.masked_fill((freq_masked | time_masked) & in_valid, 0.0)
