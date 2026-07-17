# tests/slices/test_frame_token_sampler.py
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
