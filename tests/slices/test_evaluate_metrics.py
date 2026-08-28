from src.slices.Evaluate.Metrics import char_errors, corpus_cer, corpus_wer, word_errors


def test_corpus_wer_known_value():
    refs = ["THE CAT SAT", "HELLO WORLD"]
    hyps = ["THE CAT SAT", "HELLO WORD"]  # 1 substitution over 5 ref words
    assert abs(corpus_wer(refs, hyps) - 0.2) < 1e-6


def test_corpus_cer_zero_on_exact_match():
    refs = ["ABC", "DEF"]
    assert corpus_cer(refs, refs) == 0.0


def test_summed_counts_reproduce_the_corpus_scores():
    # The eval harness aggregates per-utterance counts (so it can show a running WER and score a
    # whole weight grid without re-aligning). That is only legitimate if summing them lands exactly
    # on the corpus score -- this is the lock on that identity.
    refs = ["THE CAT SAT ON THE MAT", "HELLO WORLD", "A B C D"]
    hyps = ["THE CAT SAT ON MAT", "HELLO BRAVE WORLD", ""]
    we = [word_errors(r, h) for r, h in zip(refs, hyps)]
    ce = [char_errors(r, h) for r, h in zip(refs, hyps)]
    assert abs(sum(e for e, _ in we) / sum(n for _, n in we) - corpus_wer(refs, hyps)) < 1e-12
    assert abs(sum(e for e, _ in ce) / sum(n for _, n in ce) - corpus_cer(refs, hyps)) < 1e-12


def test_reference_length_survives_an_empty_hypothesis():
    # An empty hypothesis must still contribute its reference length to the divisor, or dropping a
    # decode would look like an improvement.
    assert word_errors("ONE TWO THREE", "") == (3, 3)


def test_sweep_selects_a_length_bonus_that_fixes_deletions():
    # RNN-T acoustic scores are un-normalised sums, so a short hypothesis wins on score even when
    # it deletes words. A positive length_bonus must be selectable, and the sweep must key its
    # results by all three weights.
    from src.slices.Decode.StreamingDecode_Response import NbestEntry
    from src.slices.Evaluate.evaluate import _sweep

    # hyp 0: 1 token, best acoustic, 2 deletions. hyp 1: 3 tokens, worse acoustic, correct.
    nbest = [
        NbestEntry(ids=[1], acoustic=-1.0, lm=0.0, ilm=0.0),
        NbestEntry(ids=[1, 2, 3], acoustic=-2.0, lm=0.0, ilm=0.0),
    ]
    prepared = [(nbest, [2, 0], 3)]

    wer = _sweep(prepared, [0.0], [0.0], [0.0, 1.0])
    assert set(wer) == {(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)}
    assert wer[(0.0, 0.0, 0.0)] == 2 / 3  # no bonus -> the short hypothesis wins
    assert wer[(0.0, 0.0, 1.0)] == 0.0  # +1.0/token -> -2+3 = 1.0 beats -1+1 = 0.0
