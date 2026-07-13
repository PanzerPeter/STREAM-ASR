# wav2vec-style span masking on input log-mel for BEST-RQ pretraining: sample span
# starts at mask_prob, extend each by mask_span, and overwrite masked frames with Gaussian
# noise. The encoder sees the masked input while the quantizer labels come from the clean
# input, so the model must infer masked content from context (SP4).
import torch


def apply_span_mask(
    features: torch.Tensor,
    lengths: torch.Tensor,
    mask_prob: float,
    mask_span: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, _ = features.shape
    mask = torch.zeros(b, t, dtype=torch.bool, device=features.device)
    for i in range(b):
        valid = int(lengths[i])
        num_spans = max(1, int(mask_prob * valid))
        starts = torch.randint(0, max(1, valid - mask_span), (num_spans,))
        for s in starts.tolist():
            mask[i, s : min(s + mask_span, valid)] = True
    masked = features.clone()
    noise = torch.randn_like(features) * noise_std
    masked[mask] = noise[mask]
    return masked, mask
