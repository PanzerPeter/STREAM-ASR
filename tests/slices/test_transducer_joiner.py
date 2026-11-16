import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner

_DE = 384


def _inputs(seed: int, t: int = 5, u1: int = 4):
    torch.manual_seed(seed)
    return torch.randn(2, t, _DE), torch.randn(2, u1, get_config().transducer.predictor_dim)


def test_forward_lattice_shape_is_padded_to_a_tensor_core_alignment():
    j = TransducerJoiner(_DE).eval()
    V = get_config().model.logits_width
    out = j(*_inputs(0))
    assert out.shape == (2, 5, 4, V + (-V % 8)) and out.shape[3] % 8 == 0


def test_padded_readout_columns_carry_exactly_zero_probability():
    j = TransducerJoiner(_DE).eval()
    V = get_config().model.logits_width
    with torch.no_grad():
        out = j(*_inputs(2, 3, 2))
    assert torch.all(torch.isneginf(out[..., V:]))
    assert torch.all(out.softmax(-1)[..., V:] == 0.0)


def test_band_cells_equal_the_full_lattice_cells():
    # The band evaluates exactly the full lattice's cells at u = lo + j, nothing else.
    j = TransducerJoiner(_DE).eval()
    enc, pred = _inputs(1, t=6, u1=7)
    lo = torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 1, 2, 2, 4]])
    cols = lo.unsqueeze(2) + torch.arange(3)
    with torch.no_grad():
        full = j(enc, pred)
        band = j(enc, pred, cols)
    expected = full.gather(2, cols.unsqueeze(3).expand(-1, -1, -1, full.shape[3]))
    assert torch.equal(band, expected)


def test_step_matches_full_lattice():
    j = TransducerJoiner(_DE).eval()
    V = get_config().model.logits_width
    enc, pred = _inputs(1, 3, 2)
    with torch.no_grad():
        full = j(enc[:1], pred[:1])
        cell = j.step(enc[:1, 1], pred[:1, 0])
    assert cell.shape == (1, V)
    assert torch.allclose(full[:, 1, 0, :V], cell, atol=1e-5)
