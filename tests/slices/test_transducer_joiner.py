import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner


def test_forward_lattice_shape():
    torch.manual_seed(0)
    j = TransducerJoiner().eval()
    De = get_config().model.encoder_dims[-1]
    Dp = get_config().transducer.predictor_dim
    V = get_config().model.logits_width
    enc = torch.randn(2, 5, De)
    pred = torch.randn(2, 4, Dp)
    out = j(enc, pred)
    assert out.shape == (2, 5, 4, V)


def test_step_matches_full_lattice():
    torch.manual_seed(1)
    j = TransducerJoiner().eval()
    De = get_config().model.encoder_dims[-1]
    Dp = get_config().transducer.predictor_dim
    enc = torch.randn(1, 3, De)
    pred = torch.randn(1, 2, Dp)
    with torch.no_grad():
        full = j(enc, pred)  # [1, 3, 2, V]
        cell = j.step(enc[:, 1], pred[:, 0])  # [1, V]
    assert torch.allclose(full[:, 1, 0], cell, atol=1e-5)
