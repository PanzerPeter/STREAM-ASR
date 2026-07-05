import torch
from src.slices.ExtractFeatures.SpecAugment_Transform import apply_spec_augment


def test_spec_augment_preserves_shape_and_masks():
    torch.manual_seed(0)
    mel = torch.randn(200, 80)
    out = apply_spec_augment(mel)
    assert out.shape == mel.shape
    assert (out == 0).any()  # at least one masked bin
