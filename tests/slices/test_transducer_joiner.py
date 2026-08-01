import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner


def test_forward_lattice_shape_is_padded_to_a_tensor_core_alignment():
    torch.manual_seed(0)
    j = TransducerJoiner().eval()
    De = get_config().model.encoder_dims[-1]
    Dp = get_config().transducer.predictor_dim
    V = get_config().model.logits_width
    enc = torch.randn(2, 5, De)
    pred = torch.randn(2, 4, Dp)
    out = j(enc, pred)
    assert out.shape == (2, 5, 4, V + (-V % 8))
    assert out.shape[3] % 8 == 0


def test_padded_readout_columns_carry_exactly_zero_probability():
    # The readout is widened only to align the lattice GEMM. The extra columns must be inert: -inf
    # logits, hence exactly-zero softmax mass, so every log-softmax, gather and gradient downstream
    # is the unpadded result. Anything finite there would silently steal probability from the vocab.
    torch.manual_seed(2)
    j = TransducerJoiner().eval()
    V = get_config().model.logits_width
    enc = torch.randn(2, 3, get_config().model.encoder_dims[-1])
    pred = torch.randn(2, 2, get_config().transducer.predictor_dim)
    with torch.no_grad():
        out = j(enc, pred)
    assert torch.all(torch.isneginf(out[..., V:]))
    probs = out.softmax(-1)
    assert torch.all(probs[..., V:] == 0.0)  # exactly, not approximately
    assert torch.allclose(probs[..., :V], out[..., :V].softmax(-1))


def test_step_matches_full_lattice():
    torch.manual_seed(1)
    j = TransducerJoiner().eval()
    De = get_config().model.encoder_dims[-1]
    Dp = get_config().transducer.predictor_dim
    V = get_config().model.logits_width
    enc = torch.randn(1, 3, De)
    pred = torch.randn(1, 2, Dp)
    with torch.no_grad():
        full = j(enc, pred)  # [1, 3, 2, V + pad]
        cell = j.step(enc[:, 1], pred[:, 0])  # [1, V] -- decode never pays for the padding
    assert cell.shape == (1, V)
    assert torch.allclose(full[:, 1, 0, :V], cell, atol=1e-5)
