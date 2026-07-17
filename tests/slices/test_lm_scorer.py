import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.Decode.LmScorer import LmScorer


def test_weight_scales_sequence_score_and_alpha_zero_is_zero():
    torch.manual_seed(0)
    model = StreamLmModel().eval()
    ids = [4, 9, 15]
    raw = LmScorer(model, weight=1.0).raw_sequence_logprob(ids)
    # weight=1 sequence_score == the unweighted raw logprob (the value alpha tuning sweeps over).
    assert abs(LmScorer(model, weight=1.0).sequence_score(ids) - raw) < 1e-5
    # alpha=0 makes the LM contribution exactly zero -> the rescored ranking is pure-acoustic.
    assert LmScorer(model, weight=0.0).sequence_score(ids) == 0.0
    # sequence_score is linear in the fusion weight; raw_sequence_logprob ignores it.
    assert abs(LmScorer(model, weight=0.5).sequence_score(ids) - 0.5 * raw) < 1e-5
    assert LmScorer(model, weight=0.5).raw_sequence_logprob(ids) == raw
