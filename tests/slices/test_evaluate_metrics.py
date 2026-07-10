from src.slices.Evaluate.Metrics import corpus_wer, corpus_cer


def test_corpus_wer_known_value():
    refs = ["THE CAT SAT", "HELLO WORLD"]
    hyps = ["THE CAT SAT", "HELLO WORD"]  # 1 substitution over 5 ref words
    assert abs(corpus_wer(refs, hyps) - 0.2) < 1e-6


def test_corpus_cer_zero_on_exact_match():
    refs = ["ABC", "DEF"]
    assert corpus_cer(refs, refs) == 0.0
