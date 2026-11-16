import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch

_FILL = torch.full((80,), -5.0)  # stands in for the raw-log-mel CMVN mean


def test_masks_with_the_cmvn_mean_and_preserves_shape():
    torch.manual_seed(0)
    features = torch.ones(2, 400, 80)
    out = apply_spec_augment_batch(features, torch.tensor([400, 300]), _FILL)
    assert out.shape == features.shape
    # Every masked value is the fill, never a raw 0.0 (= +1.5 sigma after CMVN).
    assert ((out == 1.0) | (out == -5.0)).all() and (out == -5.0).any()
    assert torch.equal(features, torch.ones(2, 400, 80))  # input not mutated


def test_padding_is_never_masked():
    torch.manual_seed(1)
    features = torch.ones(1, 40, 80)
    out = apply_spec_augment_batch(features, torch.tensor([10]), _FILL)
    assert bool((out[0, 10:, :] == 1.0).all())


def test_time_masks_stay_under_their_total_fraction():
    # x2.5 time masking is up to 25 masks, bounded together at 37.5 % of length.
    cfg = get_config().augment
    for seed in range(20):
        torch.manual_seed(seed)
        out = apply_spec_augment_batch(torch.ones(1, 1000, 80), torch.tensor([1000]), _FILL)
        masked_frames = int((out[0] == -5.0).all(dim=-1).sum())
        assert masked_frames <= cfg.specaug_max_time_fraction * 1000
