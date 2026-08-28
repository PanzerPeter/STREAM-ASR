import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand
from src.slices.ExtractFeatures.PrecomputeFeatures_Handler import precompute_features


def _write_corpus(tmp_path: Path, name: str, num_utts: int) -> tuple[Path, list[str]]:
    manifest = tmp_path / name
    paths = []
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(num_utts):
            p = tmp_path / f"{manifest.stem}-u{i}.flac"
            sf.write(p, np.random.RandomState(i).randn(4000 + i * 800).astype(np.float32), 16000)
            paths.append(str(p))
            f.write(
                json.dumps(
                    {
                        "uttid": f"u{i}",
                        "audio_filepath": str(p),
                        "text": "X",
                        "num_samples": 4000 + i * 800,
                    }
                )
                + "\n"
            )
    return manifest, paths


def test_precompute_matches_online(tmp_path: Path):
    manifest, paths = _write_corpus(tmp_path, "m.jsonl", 3)
    n = precompute_features(
        PrecomputeFeaturesCommand(str(manifest), "toy", str(tmp_path), num_workers=2)
    )
    assert n == 3
    reader = FeatureCacheReader(str(tmp_path), "toy")
    for i, p in enumerate(paths):
        online = compute_log_mel(load_audio(p))
        assert reader[i].shape == online.shape
        assert (
            reader[i] - online
        ).abs().max() < 5e-2  # fp16 tolerance at log-mel magnitude, row order preserved


def test_precompute_skips_a_complete_split(tmp_path: Path):
    # Deleting the audio is the assertion: a second pass that decoded anything could not survive it.
    manifest, paths = _write_corpus(tmp_path, "m.jsonl", 3)
    cmd = PrecomputeFeaturesCommand(str(manifest), "toy", str(tmp_path), num_workers=2)
    assert precompute_features(cmd) == 3
    before = (tmp_path / "toy.f16").read_bytes()
    for p in paths:
        os.remove(p)

    assert precompute_features(cmd) == 3
    assert (tmp_path / "toy.f16").read_bytes() == before


def test_precompute_rebuilds_a_truncated_split(tmp_path: Path):
    manifest, _ = _write_corpus(tmp_path, "m.jsonl", 3)
    cmd = PrecomputeFeaturesCommand(str(manifest), "toy", str(tmp_path), num_workers=2)
    assert precompute_features(cmd) == 3
    flat = tmp_path / "toy.f16"
    full = flat.stat().st_size
    os.truncate(flat, full // 2)  # the state a killed pass leaves behind, under a stale header

    assert precompute_features(cmd) == 3
    assert flat.stat().st_size == full
    assert FeatureCacheReader(str(tmp_path), "toy")[2].shape[0] > 0


def test_precompute_rebuilds_for_a_different_manifest(tmp_path: Path):
    # Same split name, different rows: the fingerprint must beat the completeness shortcut, or the
    # skip would silently keep a cache whose row order belongs to another manifest.
    first, _ = _write_corpus(tmp_path, "first.jsonl", 3)
    second, _ = _write_corpus(tmp_path, "second.jsonl", 4)
    assert (
        precompute_features(
            PrecomputeFeaturesCommand(str(first), "toy", str(tmp_path), num_workers=2)
        )
        == 3
    )
    assert (
        precompute_features(
            PrecomputeFeaturesCommand(str(second), "toy", str(tmp_path), num_workers=2)
        )
        == 4
    )
    assert len(FeatureCacheReader(str(tmp_path), "toy")) == 4
