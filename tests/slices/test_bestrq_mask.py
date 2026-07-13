import torch

from src.slices.PretrainEncoder.BestRqMask import apply_span_mask


def test_mask_respects_length_and_coverage():
    torch.manual_seed(0)
    feats = torch.ones(4, 100, 8)
    lengths = torch.tensor([100, 100, 60, 100])
    masked, mask = apply_span_mask(feats, lengths, mask_prob=0.1, mask_span=10, noise_std=0.1)
    assert mask.shape == (4, 100)
    assert mask.dtype == torch.bool
    # no masking beyond valid length
    assert not mask[2, 60:].any()
    # some coverage, not everything
    frac = mask[0].float().mean().item()
    assert 0.0 < frac < 0.9


def test_masked_positions_get_noise_unmasked_unchanged():
    torch.manual_seed(0)
    feats = torch.ones(2, 50, 4)
    lengths = torch.tensor([50, 50])
    masked, mask = apply_span_mask(feats, lengths, mask_prob=0.15, mask_span=8, noise_std=0.1)
    assert torch.allclose(masked[~mask], torch.ones_like(masked[~mask]))  # untouched == original
    assert not torch.allclose(masked[mask], torch.ones_like(masked[mask]))  # replaced by noise
