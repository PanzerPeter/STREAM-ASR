import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.Decode.LmScorer import LmScorer


def _sos() -> int:
    return get_config().model.sos_id


def test_weight_scales_sequence_score_and_alpha_zero_is_zero():
    torch.manual_seed(0)
    model = StreamLmModel().eval()
    ids = [4, 9, 15]
    base = LmScorer(model, weight=1.0).sequence_score(ids)
    # alpha=0 makes the LM contribution exactly zero -> the decoder is byte-identical to pre-LM.
    assert LmScorer(model, weight=0.0).sequence_score(ids) == 0.0
    # sequence_score is linear in the fusion weight.
    assert abs(LmScorer(model, weight=0.5).sequence_score(ids) - 0.5 * base) < 1e-5


def test_step_score_scales_by_weight_and_threads_state():
    torch.manual_seed(0)
    model = StreamLmModel().eval()
    logp_full, state = LmScorer(model, weight=1.0).step_score(_sos(), None)
    logp_half, _ = LmScorer(model, weight=0.5).step_score(_sos(), None)
    torch.testing.assert_close(logp_half, 0.5 * logp_full, atol=1e-5, rtol=1e-5)
    assert state is not None  # LM state is carried forward for the next token
