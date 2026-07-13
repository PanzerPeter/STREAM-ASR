import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand
from src.slices.ComputeCmvn.ComputeCmvn_Handler import compute_cmvn


def _fixture(root: Path, n: int) -> str:
    manifest = root / "m.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(n):
            p = root / f"u{i}.flac"
            sf.write(p, np.random.RandomState(i).randn(4000).astype(np.float32), 16000)
            f.write(json.dumps({"audio_filepath": str(p), "text": "X", "num_samples": 4000}) + "\n")
    return str(manifest)


def test_cmvn_sampled_is_deterministic(tmp_path):
    manifest = _fixture(tmp_path, 10)
    out = tmp_path / "cmvn.pt"
    a = compute_cmvn(ComputeCmvnCommand(manifest, str(out), sample_frac=0.5, seed=1))
    b = compute_cmvn(ComputeCmvnCommand(manifest, str(out), sample_frac=0.5, seed=1))
    assert a["mean"].shape == (80,) and a["std"].shape == (80,)
    assert bool((a["mean"] == b["mean"]).all())  # same seed -> same sample -> same stats
    assert bool(np.isfinite(a["mean"].numpy()).all())
    assert bool((a["std"] > 0).all())  # variance-floor invariant (audio.cmvn_eps clamp)

    # Different seeds at the same sample_frac must draw different subsets of the
    # manifest, and therefore produce different stats. If the sample_frac branch
    # were ever skipped (k collapses to len(rows)), a/b/c would be identical
    # despite the seed change, and this assertion would catch it.
    c = compute_cmvn(ComputeCmvnCommand(manifest, str(out), sample_frac=0.5, seed=2))
    assert not bool((a["mean"] == c["mean"]).all())  # different seed -> different sample

    # The handler persists exactly the returned stats dict via torch.save; round-trip
    # it to confirm the on-disk artifact matches what callers of compute_cmvn get back.
    loaded = torch.load(out, weights_only=True)
    assert set(loaded.keys()) == {"mean", "std"}
    assert bool((loaded["mean"] == c["mean"]).all())
    assert bool((loaded["std"] == c["std"]).all())
