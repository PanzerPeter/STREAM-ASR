from src.slices.TrainAcousticModel.TransducerTrainer_Handler import _probe_batches


def test_probe_spans_both_ends_of_the_duration_sorted_range():
    # The dev loader is duration-sorted, so a head slice probes only the shortest clips (on the real
    # dev manifest: nothing above 2.6 s against a 32.6 s maximum). The probe must reach the last
    # batch as well as the first, or long-utterance regressions are invisible to it.
    sel = _probe_batches(n_batches=114, n_utts=2703, wer_utts=200)
    assert min(sel) == 0
    assert max(sel) == 113


def test_probe_size_tracks_the_utterance_budget():
    # k is the budget converted to batches at the loader's mean batch size (2703/114 ~= 23.7 utts),
    # so 200 utterances is ~8 batches and doubling the budget roughly doubles the count.
    assert len(_probe_batches(114, 2703, 200)) == 8
    assert len(_probe_batches(114, 2703, 400)) == 17


def test_probe_batches_are_evenly_spaced():
    gaps = sorted(sel := _probe_batches(100, 1000, 100))
    steps = {b - a for a, b in zip(gaps, gaps[1:])}
    assert len(sel) == 10
    assert max(steps) - min(steps) <= 1  # even placement, off-by-one from rounding only


def test_probe_degenerates_safely():
    # Budget larger than the loader, and a single-batch loader: never empty, never out of range.
    assert _probe_batches(3, 10, 10_000) == {0, 1, 2}
    assert _probe_batches(1, 10, 200) == {0}
    assert _probe_batches(50, 10_000, 1) == {0}
