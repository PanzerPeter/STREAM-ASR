import torch

from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch


def test_batch_specaugment_masks_and_preserves_shape():
    torch.manual_seed(0)
    features = torch.ones(2, 50, 80)
    lengths = torch.tensor([50, 30])
    out = apply_spec_augment_batch(features, lengths)
    assert out.shape == features.shape
    assert (out == 0.0).any()  # something was masked
    assert not torch.equal(out, features)  # input not mutated in place
    assert torch.equal(features, torch.ones(2, 50, 80))


def test_batch_specaugment_respects_length():
    torch.manual_seed(1)
    features = torch.ones(1, 40, 80)
    lengths = torch.tensor([10])
    out = apply_spec_augment_batch(features, lengths)
    assert bool(
        (out[0, 10:, :] == 1.0).all()
    )  # padding region (t >= length) untouched by time masks
