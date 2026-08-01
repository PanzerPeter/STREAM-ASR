import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.shared_kernel.Config_Adapter import get_config


def _lm():
    torch.manual_seed(0)
    return StreamLmModel().eval()


def test_forward_shape_matches_vocab():
    lm = _lm()
    v = get_config().model.decoder_vocab_size
    out = lm(torch.randint(0, v, (2, 7)))
    assert out.shape == (2, 7, v)


def test_output_weight_is_tied_to_embedding():
    lm = _lm()
    assert lm.head.weight is lm.tok_emb.weight


def test_causality_end_to_end():
    lm = _lm()
    v = get_config().model.decoder_vocab_size
    x = torch.randint(0, v, (1, 8))
    a = lm(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randint(0, v, (1, 3))
    b = lm(x2)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-5, rtol=1e-5)
