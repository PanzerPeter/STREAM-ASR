import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.Decode.LmScorer import LmScorer


def test_weight_scales_sequence_score_and_alpha_zero_is_zero():
    torch.manual_seed(0)
    model = StreamLmModel().eval()
    nbest = [[4, 9, 15], [4, 9]]
    raw = LmScorer(model, weight=1.0).raw_sequence_logprobs(nbest)
    # weight=1 scores == the unweighted raw logprobs (the values alpha tuning sweeps over).
    for scored, r in zip(LmScorer(model, weight=1.0).sequence_scores(nbest), raw):
        assert abs(scored - r) < 1e-5
    # alpha=0 makes the LM contribution exactly zero -> the rescored ranking is pure-acoustic.
    assert LmScorer(model, weight=0.0).sequence_scores(nbest) == [0.0, 0.0]
    # Scores are linear in the fusion weight; raw_sequence_logprobs ignores it.
    for scored, r in zip(LmScorer(model, weight=0.5).sequence_scores(nbest), raw):
        assert abs(scored - 0.5 * r) < 1e-5
    assert LmScorer(model, weight=0.5).raw_sequence_logprobs(nbest) == raw
