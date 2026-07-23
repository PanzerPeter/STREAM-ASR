import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.Decode.InternalLmScorer import InternalLmScorer


def _model() -> TransducerModel:
    torch.manual_seed(0)
    return TransducerModel().eval()


def test_score_is_independent_of_encoder_memory():
    # The internal LM is the transducer's prior with the acoustic evidence removed, so its score
    # must not move when the encoder memory does -- that independence is the whole definition.
    scorer = InternalLmScorer(_model())
    assert scorer.sequence_logprob([4, 9, 15]) == scorer.sequence_logprob([4, 9, 15])


def test_score_is_a_proper_log_probability_over_non_blank_labels():
    # Renormalising over the non-blank labels must yield a distribution: the exp of the one-token
    # scores over the whole vocab sums to 1, and every score is <= 0.
    scorer = InternalLmScorer(_model())
    vocab = get_config().model.vocab_size
    total = sum(pow(2.718281828459045, scorer.sequence_logprob([t])) for t in range(vocab))
    assert abs(total - 1.0) < 1e-3
    assert scorer.sequence_logprob([4, 9, 15]) <= 0.0


def test_empty_hypothesis_scores_zero():
    assert InternalLmScorer(_model()).sequence_logprob([]) == 0.0


def test_batch_matches_per_sequence():
    scorer = InternalLmScorer(_model())
    seqs = [[5], [3, 7, 42], [], [11, 12]]
    batched = scorer.sequence_logprob_batch(seqs)
    for b, s in zip(batched, seqs):
        assert abs(b - scorer.sequence_logprob(s)) < 1e-3
