import pytest
import torch

from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


@pytest.mark.slow
def test_transducer_overfits_one_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    batch = collate_features(
        [(torch.randn(160, 80), [3, 4, 5, 6, 7]), (torch.randn(120, 80), [8, 9, 10])]
    )
    model = TransducerModel(cmvn_path=None).to(device).train()
    # Overfit a FIXED batch: masking would move the target each step. cr_ctc has to go too -- its
    # two-view path SpecAugments unconditionally, so _spec_augment alone does not fix the input.
    model._spec_augment = False
    model._cr_ctc = False
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        total, *_ = model.joint_loss(batch, chunk_size=0)
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < 0.5 * first
