import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader, write_feature_cache
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset


class _Tok:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 7 for c in text]


def test_dataset_reads_cache(tmp_path: Path):
    mel0 = np.random.randn(6, 80).astype(np.float32)
    write_feature_cache(str(tmp_path), "toy", [mel0, np.random.randn(4, 80).astype(np.float32)])
    manifest = tmp_path / "m.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for t in ("AB", "CD"):
            f.write(json.dumps({"audio_filepath": "unused", "text": t, "num_samples": 1000}) + "\n")
    ds = LibriSpeechDataset(str(manifest), _Tok(), cache=FeatureCacheReader(str(tmp_path), "toy"))
    mel, ids = ds[0]
    assert mel.shape == (6, 80)
    assert torch.allclose(mel, torch.from_numpy(mel0), atol=1e-2)  # cache read, no augmentation
    assert ids == [ord("A") % 7, ord("B") % 7]


def _wav_manifest(tmp_path: Path, speed: float) -> str:
    sr = 16000
    sf.write(tmp_path / "u.flac", np.random.randn(sr).astype(np.float32), sr)
    manifest = tmp_path / f"m_{speed}.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "audio_filepath": str(tmp_path / "u.flac"),
                    "text": "HI",
                    "num_samples": sr,
                    "speed": speed,
                }
            )
            + "\n"
        )
    return str(manifest)


def test_speed_perturb_slows_lengthens_mel(tmp_path: Path):
    # 0.9x speed lengthens the audio, so its log-mel has strictly more frames than the 1.0x version.
    ds_ref = LibriSpeechDataset(_wav_manifest(tmp_path, 1.0), _Tok())
    ds_slow = LibriSpeechDataset(_wav_manifest(tmp_path, 0.9), _Tok())
    assert ds_slow[0][0].shape[0] > ds_ref[0][0].shape[0]


def test_speed_perturb_row_with_cache_is_rejected(tmp_path: Path):
    # Perturbed rows have no precomputed features; pairing them with a cache must fail loudly, not
    # silently return the clean cached mel under a perturbed label.
    write_feature_cache(str(tmp_path), "toy", [np.random.randn(6, 80).astype(np.float32)])
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": "x", "text": "HI", "num_samples": 9000, "speed": 0.9}) + "\n",
        encoding="utf-8",
    )
    ds = LibriSpeechDataset(str(manifest), _Tok(), cache=FeatureCacheReader(str(tmp_path), "toy"))
    with pytest.raises(ValueError):
        _ = ds[0]
