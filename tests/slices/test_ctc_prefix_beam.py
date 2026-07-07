import torch
from src.slices.Decode.CtcPrefixBeam import CtcPrefixBeam


def _one_hot_logprobs(seq, vocab):
    lp = torch.full((len(seq), vocab), -20.0)
    for t, idx in enumerate(seq):
        lp[t, idx] = 0.0
    return lp.log_softmax(dim=-1)


def test_collapses_repeats_and_blanks():
    blank = 3
    beam = CtcPrefixBeam(blank_id=blank, beam_size=8)
    beam.reset()
    beam.advance(_one_hot_logprobs([0, 0, blank, 0], vocab=4))  # a a _ a -> (a, a)
    assert beam.nbest()[0][0] == (0, 0)


def test_partial_available_across_chunks():
    beam = CtcPrefixBeam(blank_id=3, beam_size=8)
    beam.reset()
    beam.advance(_one_hot_logprobs([0], vocab=4))
    assert beam.partial() == [0]
    beam.advance(_one_hot_logprobs([1], vocab=4))
    assert beam.partial() == [0, 1]


def test_nbest_ranks_by_probability():
    # Non-vacuous: assert the returned scores actually reflect the input log-probs, not merely
    # that nbest() sorts its own output. One frame, token 0 > 1 > 2, must rank (0,)>(1,)>(2,).
    beam = CtcPrefixBeam(blank_id=3, beam_size=8)
    beam.reset()
    logits = torch.tensor([[2.0, 1.0, 0.0, 0.0]])  # blank (id 3) is the lowest
    beam.advance(logits.log_softmax(dim=-1))
    ranked = beam.nbest()
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)  # returned best-first
    d = dict(ranked)
    assert ranked[0][0] == (0,)  # highest-probability single token tops the list
    assert d[(0,)] > d[(1,)] > d[(2,)]  # ordering reflects real log-probs
