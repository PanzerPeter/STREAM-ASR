import torch
import torch.nn as nn

from src.slices.Decode.LmScorer import LmScorer
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


class _WordTok:
    # One id per word (stable across calls), enough to exercise the text -> ids -> score path.
    def encode(self, text: str) -> list[int]:
        return [2 + sum(map(ord, w)) % 100 for w in text.split()]


def _model() -> StreamLmModel:
    torch.manual_seed(0)
    model = StreamLmModel().eval()
    nn.init.normal_(model.head.weight)
    return model


def test_weight_scales_sequence_score_and_alpha_zero_is_zero():
    model, tok = _model(), _WordTok()
    nbest = ["THE CAT SAT", "THE CAT"]
    raw = LmScorer(model, tok, weight=1.0).raw_sequence_logprobs(nbest)
    # weight=1 scores == the unweighted raw logprobs (the values alpha tuning sweeps over).
    for scored, r in zip(LmScorer(model, tok, weight=1.0).sequence_scores(nbest), raw):
        assert abs(scored - r) < 1e-5
    # alpha=0 makes the LM contribution exactly zero -> the rescored ranking is pure-acoustic.
    assert LmScorer(model, tok, weight=0.0).sequence_scores(nbest) == [0.0, 0.0]
    # Scores are linear in the fusion weight; raw_sequence_logprobs ignores it.
    for scored, r in zip(LmScorer(model, tok, weight=0.5).sequence_scores(nbest), raw):
        assert abs(scored - 0.5 * r) < 1e-5
    assert LmScorer(model, tok, weight=0.5).raw_sequence_logprobs(nbest) == raw


def test_text_is_scored_through_the_lm_tokenizer():
    # The LM scores hypothesis text re-encoded with its own tokenizer, so the score is the model's
    # sequence log-probability of exactly those ids.
    model, tok = _model(), _WordTok()
    text = "THE CAT SAT"
    expected = model.sequence_logprob(tok.encode(text))
    assert abs(LmScorer(model, tok, weight=1.0).raw_sequence_logprobs([text])[0] - expected) < 1e-4
