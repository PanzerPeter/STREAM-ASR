import math

import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


def _lm() -> StreamLmModel:
    torch.manual_seed(0)
    return StreamLmModel().eval()


def test_forward_shape_matches_lm_vocab():
    lm = _lm()
    v = get_config().lm.vocab_size
    out = lm(torch.randint(0, v, (2, 7)))
    assert out.shape == (2, 7, v)


def test_readout_is_untied_and_starts_uniform():
    # Zero-initialised readout: the step-0 loss is exactly ln V, not the hundreds of nats a readout
    # through the N(0, 1) embedding table produces.
    lm = _lm()
    assert lm.head.weight is not lm.tok_emb.weight
    v = get_config().lm.vocab_size
    x = torch.randint(0, v, (2, 9))
    with torch.no_grad():
        logits = lm(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, v), x.reshape(-1))
    assert abs(float(loss) - math.log(v)) < 1e-5


def test_value_residual_weight_is_learnable_below_layer_zero():
    lm = _lm()
    lam = get_config().lm.value_residual_lambda
    assert lm.blocks[0].attn.value_lambda is None  # layer 0 produces the residual values
    for blk in lm.blocks[1:]:
        assert isinstance(blk.attn.value_lambda, nn.Parameter)
        assert blk.attn.value_lambda.item() == lam


def test_causality_end_to_end():
    lm = _lm()
    nn.init.normal_(lm.head.weight)  # a zero readout would make this vacuous
    v = get_config().lm.vocab_size
    x = torch.randint(0, v, (1, 8))
    a = lm(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randint(0, v, (1, 3))
    b = lm(x2)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[:, 5:], b[:, 5:])
