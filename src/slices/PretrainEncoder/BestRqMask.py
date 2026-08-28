# Wav2vec-style span masking on input log-mel for BEST-RQ pretraining: sample span starts per
# frame at mask_prob, extend each by mask_span, and overwrite masked frames with noise. The encoder
# sees the masked input while the quantizer labels come from the clean input, so the model must
# infer masked content from context.
import torch
import torch.nn.functional as F


def apply_span_mask(
    features: torch.Tensor,
    lengths: torch.Tensor,
    mask_prob: float,
    mask_span: int,
    noise_std: float,
    fill_mean: torch.Tensor,
    fill_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Span-mask `features` [B, T, F] in place of a Python loop over utterances and spans.

    THE FILL IS SPECIFIED IN NORMALIZED SPACE. `noise_std` is the paper's 0.1, applied after its
    "the input data is normalized to have 0 mean and standard deviation of 1"; this repo masks the
    RAW log-mel because CMVN lives inside the encoder, so the fill is de-normalized here as
    `fill_mean + fill_std * N(0, noise_std)`. Pass the encoder's CMVN statistics. Drawing
    `N(0, 0.1)` in raw log-mel space instead -- what this did before -- lands masked frames at
    +1.46 sigma of the data with 0.024 sigma of variance (MEASURED against `data/features/cmvn.pt`:
    raw log-mel mean -5.65, std 4.06), i.e. a constant high-energy plateau rather than a signal
    that has been removed.

    Starts are sampled per frame at `mask_prob`, so the masked fraction is 1-(1-p)^span before edge
    effects, and a span is expanded by a left-window max: frame t is masked iff a start fell in
    [t - span + 1, t]. The previous implementation drew `int(mask_prob * valid)` starts per
    utterance and wrote each span with its own indexed assignment -- ~1,200 Python iterations and
    ~1,200 single-slice CUDA writes per step at B=16, T=1500 -- for the same distribution.
    """
    b, t, _ = features.shape
    device = features.device
    lengths = lengths.to(device)
    positions = torch.arange(t, device=device)[None, :]
    # A span may only start where it still fits, which is also what keeps starts off the padding.
    last_start = (lengths - mask_span).clamp(min=1)[:, None]
    starts = (torch.rand(b, t, device=device) < mask_prob) & (positions < last_start)
    # An utterance that drew no start contributes nothing to the loss, so force one rather than
    # spend its frames. Computed unconditionally: branching on `starts.any()` costs a device sync.
    forced = (torch.rand(b, device=device) * last_start.squeeze(1)).long()
    starts[torch.arange(b, device=device), forced] |= ~starts.any(dim=1)

    covered = F.max_pool1d(
        F.pad(starts.to(features.dtype).unsqueeze(1), (mask_span - 1, 0)),
        kernel_size=mask_span,
        stride=1,
    ).squeeze(1)
    mask = (covered > 0) & (positions < lengths[:, None])

    noise = fill_mean + fill_std * (torch.randn_like(features) * noise_std)
    return torch.where(mask.unsqueeze(-1), noise, features), mask
