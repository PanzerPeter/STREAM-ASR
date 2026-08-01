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


def test_token_window_sort_groups_similar_transcript_lengths(tmp_path):
    # 40 utterances of one duration with alternating transcript lengths. Sorting by duration alone
    # leaves the long/short alternation intact, so every batch is charged at the long U; the window
    # re-sort separates them, which is where the ~9% RNN-T lattice saving comes from.
    rows = [{"num_samples": 16000 + i, "text": "A" * (400 if i % 2 else 40)} for i in range(40)]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    budgets = dict(max_frames_per_batch=400, max_tokens_per_batch=880)

    unsorted_batches = FrameBucketSampler(str(p), token_sort_window=0, **budgets)._build_batches()
    windowed = FrameBucketSampler(str(p), token_sort_window=64, **budgets)._build_batches()

    def worst_u(batches):
        return sum(len(b) * max(len(rows[i]["text"]) for i in b) for b in batches)

    assert worst_u(windowed) < worst_u(unsorted_batches)


def test_token_window_sort_is_off_without_a_token_budget(tmp_path):
    # BEST-RQ pretraining passes no token budget and has no U dimension to pad, so the batch stream
    # must stay in pure duration order.
    m = _manifest(tmp_path)
    frame_only = FrameBucketSampler(m, max_frames_per_batch=100, token_sort_window=2)
    assert frame_only._order == sorted(range(3), key=lambda i: frame_only._frames[i])


def test_lattice_budget_bounds_the_worst_case_batch(tmp_path):
    # Two sum budgets never bound B * max(frames) * max(tokens), which is what the RNN-T lattice
    # (and therefore peak VRAM) actually costs. One long-transcript utterance among short ones is
    # enough to make a batch that fits both sums cost several times the typical one.
    rows = [{"num_samples": 16000, "text": "A" * 20} for _ in range(9)]
    rows.append({"num_samples": 16000, "text": "B" * 300})
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    budgets = dict(max_frames_per_batch=10_000, max_tokens_per_batch=10_000, token_sort_window=0)

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
