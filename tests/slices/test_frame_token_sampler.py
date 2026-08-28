import json

from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler


def _manifest(tmp_path):
    rows = [
        {"num_samples": 16000, "text": "A" * 200},
        {"num_samples": 16000, "text": "B" * 200},
        {"num_samples": 16000, "text": "C" * 200},
    ]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_token_budget_splits_batches(tmp_path):
    m = _manifest(tmp_path)
    # Frame budget alone would put all three in one batch; a 300-char token budget forces splits.
    s = FrameBucketSampler(m, max_frames_per_batch=10_000_000, max_tokens_per_batch=300)
    batches = list(s)
    assert all(sum(200 for _ in b) <= 300 or len(b) == 1 for b in batches)
    assert len(batches) == 3  # each 200-char utt exceeds a shared 300 budget -> one per batch


def test_default_none_is_frame_only(tmp_path):
    m = _manifest(tmp_path)
    s = FrameBucketSampler(m, max_frames_per_batch=10_000_000)
    assert len(list(s)) == 1  # all three fit the huge frame budget, no token cap


def _alternating_manifest(tmp_path, n=40, short=40, long=400):
    # Equal-duration utterances whose transcripts alternate in length -- the speaking-rate spread
    # the token sort exists to collapse.
    rows = [{"num_samples": 16000 + i, "text": "A" * (long if i % 2 else short)} for i in range(n)]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _lattice_cells(sampler):
    # What the full [B, T, U+1, V] objective is charged: B x max(frames) x max(transcript).
    return sum(
        len(b) * max(sampler._frames[i] for i in b) * max(sampler._tokens[i] for i in b)
        for b in sampler._batch_list()
    )


def test_order_defaults_to_a_pure_duration_sort(tmp_path):
    # token_sort_window defaults to 1 (off), and off must mean off: transcript length does not
    # influence the order at all, so the batch stream is the one every measured recipe used.
    p = _alternating_manifest(tmp_path)
    s = FrameBucketSampler(p, max_frames_per_batch=400, max_tokens_per_batch=880)
    assert s._order == sorted(range(40), key=lambda i: s._frames[i])


def test_token_sort_window_cuts_lattice_padding(tmp_path):
    # The full [B,T,U+1,V] lattice is charged at the batch's *max* transcript length, not its mean,
    # so mixing a 120-char utterance with 40-char ones costs 3x on every one of them. Re-sorting by
    # transcript length inside a duration window makes a batch homogeneous in U -- and every
    # utterance must still be emitted exactly once.
    p = _alternating_manifest(tmp_path, n=2000, short=40, long=120)
    budgets = dict(max_frames_per_batch=10**9, max_tokens_per_batch=480)

    off = FrameBucketSampler(p, token_sort_window=1, **budgets)
    on = FrameBucketSampler(p, token_sort_window=500, **budgets)
    assert _lattice_cells(on) < 0.75 * _lattice_cells(off)
    assert sorted(i for b in on for i in b) == list(range(2000))  # nothing lost or duplicated


def test_token_sort_window_must_exceed_the_batch_size_to_do_anything(tmp_path):
    # The footgun. The re-sort only pays off where a whole batch fits inside one homogeneous run of
    # the window; a window of the same order as the batch just interleaves the two lengths again and
    # buys nothing. On train_sp2 the mean batch is ~22 utterances, which is why the candidate values
    # are in the hundreds and 1 means off.
    p = _alternating_manifest(tmp_path, n=2000, short=40, long=120)
    budgets = dict(max_frames_per_batch=10**9, max_tokens_per_batch=480)  # ~6-12 utts per batch

    off = _lattice_cells(FrameBucketSampler(p, token_sort_window=1, **budgets))
    narrow = _lattice_cells(FrameBucketSampler(p, token_sort_window=8, **budgets))
    wide = _lattice_cells(FrameBucketSampler(p, token_sort_window=500, **budgets))
    assert narrow > 0.99 * off  # a window the size of a batch is inert
    assert wide < 0.75 * off


def test_token_sort_window_is_inert_without_a_token_budget(tmp_path):
    # The window re-sorts by a quantity no budget is tracking, so without max_tokens_per_batch it
    # would perturb the batch stream for nothing. Guard that it stays a plain duration sort.
    p = _alternating_manifest(tmp_path)
    s = FrameBucketSampler(p, max_frames_per_batch=400, token_sort_window=8)
    assert s._order == sorted(range(40), key=lambda i: s._frames[i])


def test_shuffle_permutes_without_mutating_the_cached_order(tmp_path):
    # The batch list is built once and cached; shuffling must not consume it. Two epochs must give
    # two different orders over the SAME batches, and epoch N's permutation must depend only on the
    # seed and N -- not on how many times the sampler happened to be iterated before.
    p = _alternating_manifest(tmp_path)
    s = FrameBucketSampler(p, max_frames_per_batch=400, max_tokens_per_batch=880, shuffle=True)
    canonical = [tuple(b) for b in s._batch_list()]
    first, second = [tuple(b) for b in s], [tuple(b) for b in s]
    assert sorted(first) == sorted(canonical) == sorted(second)
    assert first != second
    assert [tuple(b) for b in s._batch_list()] == canonical


def test_lattice_budget_bounds_the_worst_case_batch(tmp_path):
    # Two sum budgets never bound B * max(frames) * max(tokens), which is what the alignment grid
    # actually costs -- VRAM under the full objective, simple-loss bandwidth under the pruned one.
    # One long-transcript utterance among short ones is enough to make a batch that fits both sums
    # cost several times the typical one.
    rows = [{"num_samples": 16000, "text": "A" * 20} for _ in range(9)]
    rows.append({"num_samples": 16000, "text": "B" * 300})
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    budgets = dict(max_frames_per_batch=10_000, max_tokens_per_batch=10_000)

    def worst(sampler):
        return max(
            len(b) * max(sampler._frames[i] for i in b) * max(sampler._tokens[i] for i in b)
            for b in sampler._build_batches()
        )

    uncapped = FrameBucketSampler(str(p), **budgets)
    capped = FrameBucketSampler(str(p), max_lattice_per_batch=150_000, **budgets)
    # Both sums fit, so uncapped puts all ten in one batch charged at the 300-char
    # transcript: 10 * 100 frames * 300 chars = 300k, double what the cap allows.
    assert worst(uncapped) == 300_000
    assert worst(capped) <= 150_000
    assert sorted(i for b in capped._build_batches() for i in b) == list(range(10))  # none lost


def test_lattice_budget_still_emits_an_oversized_utterance(tmp_path):
    # A single utterance that alone exceeds the cap must still be yielded in a batch of its own,
    # not dropped and not looped on.
    rows = [{"num_samples": 16000, "text": "A" * 500}]
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(rows[0]) + "\n")
    s = FrameBucketSampler(str(p), max_frames_per_batch=10_000, max_lattice_per_batch=1)
    assert list(s) == [[0]]
