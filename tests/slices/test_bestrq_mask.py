import torch

from src.slices.PretrainEncoder.BestRqMask import apply_span_mask

_ZERO = torch.zeros(8)
_ONE = torch.ones(8)


def test_mask_respects_length_and_coverage():
    torch.manual_seed(0)
    feats = torch.ones(4, 100, 8)
    lengths = torch.tensor([100, 100, 60, 100])
    masked, mask = apply_span_mask(feats, lengths, 0.1, 10, 0.1, _ZERO, _ONE)
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
    masked, mask = apply_span_mask(feats, lengths, 0.15, 8, 0.1, torch.zeros(4), torch.ones(4))
    assert torch.allclose(masked[~mask], torch.ones_like(masked[~mask]))  # untouched == original
    assert not torch.allclose(masked[mask], torch.ones_like(masked[mask]))  # replaced by noise


def test_masked_fraction_matches_span_geometry():
    # Per-frame start probability extended by mask_span gives 1-(1-p)^span before edge effects;
    # this is the contract config/pretrain.yaml's mask_prob comment reasons from.
    torch.manual_seed(0)
    feats = torch.zeros(16, 2000, 8)
    lengths = torch.full((16,), 2000)
    _, mask = apply_span_mask(feats, lengths, 0.02, 30, 0.1, _ZERO, _ONE)
    assert abs(mask.float().mean().item() - (1 - 0.98**30)) < 0.03


def test_fill_is_noise_std_in_normalized_space():
    # The fill is specified AFTER CMVN: with these statistics the masked frames must come back
    # centred on cmvn_mean with cmvn_std * noise_std of spread, not on 0 with noise_std of spread.
    torch.manual_seed(0)
    mean, std = torch.full((8,), -5.9), torch.full((8,), 4.2)
    feats = torch.zeros(8, 4000, 8)
    lengths = torch.full((8,), 4000)
    masked, mask = apply_span_mask(feats, lengths, 0.02, 30, 0.1, mean, std)
    filled = masked[mask]
    assert abs(filled.mean().item() - (-5.9)) < 0.05
    assert abs(filled.std().item() - 4.2 * 0.1) < 0.05


def test_every_utterance_gets_at_least_one_span():
    # A row that drew no start would contribute nothing to the loss; short rows at a low
    # mask_prob are exactly where that happens.
    torch.manual_seed(0)
    feats = torch.zeros(32, 60, 8)
    lengths = torch.full((32,), 60)
    _, mask = apply_span_mask(feats, lengths, 0.001, 10, 0.1, _ZERO, _ONE)
    assert mask.any(dim=1).all()


def test_spans_are_contiguous_runs_of_mask_span():
    # One start on a row long enough that overlap is unlikely: the run it produces is mask_span
    # frames wide, i.e. the left-window expansion covers [s, s+span-1].
    torch.manual_seed(3)
    feats = torch.zeros(1, 500, 8)
    lengths = torch.tensor([500])
    _, mask = apply_span_mask(feats, lengths, 0.0, 30, 0.1, _ZERO, _ONE)
    idx = mask[0].nonzero().flatten()
    assert idx.numel() == 30
    assert (idx.diff() == 1).all()
