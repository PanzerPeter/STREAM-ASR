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
    s = FrameBucketSampler((m,), max_frames_per_batch=10_000_000, max_tokens_per_batch=300)
    batches = list(s)
    assert all(sum(200 for _ in b) <= 300 or len(b) == 1 for b in batches)
    assert len(batches) == 3  # each 200-char utt exceeds a shared 300 budget -> one per batch


def test_default_none_is_frame_only(tmp_path):
    m = _manifest(tmp_path)
    s = FrameBucketSampler((m,), max_frames_per_batch=10_000_000)
    assert len(list(s)) == 1  # all three fit the huge frame budget, no token cap


def _alternating_manifest(tmp_path, n=40, short=40, long=400):
    # Equal-duration utterances whose transcripts alternate in length -- the speaking-rate spread
    # the token sort exists to collapse.
    rows = [{"num_samples": 16000 + i, "text": "A" * (long if i % 2 else short)} for i in range(n)]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_order_is_a_pure_duration_sort(tmp_path):
    # Transcript length never influences the order: with the band the lattice no longer scales
    # with U, so there is no U padding to sort away.
    p = _alternating_manifest(tmp_path)
    s = FrameBucketSampler((p,), max_frames_per_batch=400, max_tokens_per_batch=880)
    assert s._order == sorted(range(40), key=lambda i: s._frames[i])


def test_shuffle_permutes_without_mutating_the_cached_order(tmp_path):
    # The batch list is built once and cached; shuffling must not consume it. Two epochs must give
    # two different orders over the SAME batches, and epoch N's permutation must depend only on the
    # seed and N -- not on how many times the sampler happened to be iterated before.
    p = _alternating_manifest(tmp_path)
    s = FrameBucketSampler((p,), max_frames_per_batch=400, max_tokens_per_batch=880, shuffle=True)
    canonical = [tuple(b) for b in s._batch_list()]
    first, second = [tuple(b) for b in s], [tuple(b) for b in s]
    assert sorted(first) == sorted(canonical) == sorted(second)
    assert first != second
    assert [tuple(b) for b in s._batch_list()] == canonical


def test_lattice_budget_bounds_the_worst_case_batch(tmp_path):
    # Two sum budgets never bound B * max(frames) * max(tokens), which is what the alignment grid
    # actually costs: the full RNN-T lattice's VRAM.
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

    uncapped = FrameBucketSampler((str(p),), **budgets)
    capped = FrameBucketSampler((str(p),), max_lattice_per_batch=150_000, **budgets)
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
    s = FrameBucketSampler((str(p),), max_frames_per_batch=10_000, max_lattice_per_batch=1)
    assert list(s) == [[0]]


def test_several_manifests_index_like_their_concat_dataset(tmp_path):
    # The trainer reads one LibriSpeechDataset per source under a ConcatDataset and one sampler
    # over all their manifests. Index i must name the same row in both, or a batch is bucketed on
    # one clip's length and trained on another's audio.
    import numpy as np
    from torch.utils.data import ConcatDataset

    from src.shared_kernel.Config_Adapter import get_config
    from src.slices.ExtractFeatures.FeatureCache import (
        FeatureCacheReader,
        manifest_fingerprint,
        write_feature_cache,
    )
    from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset

    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    hop, n_mels = get_config().audio.hop_length, get_config().audio.n_mels
    datasets, manifests = [], []
    for name, frames in (("ls", [30, 10, 50]), ("cv", [20, 40])):
        path = tmp_path / f"{name}.jsonl"
        rows = [
            {"uttid": f"{name}{i}", "text": "X" * (i + 1), "num_samples": f * hop}
            for i, f in enumerate(frames)
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        fp = manifest_fingerprint(str(path))
        write_feature_cache(str(tmp_path), name, (np.zeros((f, n_mels)) for f in frames), fp)
        datasets.append(
            LibriSpeechDataset(str(path), _Tok(), FeatureCacheReader(str(tmp_path), name, fp))
        )
        manifests.append(str(path))

    data = ConcatDataset(datasets)
    sampler = FrameBucketSampler(tuple(manifests), max_frames_per_batch=60, shuffle=True, seed=3)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(5))
    for i in seen:
        mel, _ = data[i]
        assert mel.shape[0] == sampler._frames[i]
