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
    from src.slices.ExtractFeatures.FeatureCache import manifest_fingerprint

    mel0 = np.random.randn(6, 80).astype(np.float32)
    manifest = tmp_path / "m.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for i, t in enumerate(("AB", "CD")):
            f.write(
                json.dumps(
                    {
                        "uttid": f"u{i}",
                        "audio_filepath": "unused",
                        "text": t,
                        "num_samples": 1000,
                    }
                )
                + "\n"
            )
    write_feature_cache(
        str(tmp_path),
        "toy",
        [mel0, np.random.randn(4, 80).astype(np.float32)],
        manifest_fingerprint(str(manifest)),
    )
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


def test_dataset_rejects_a_cache_built_for_another_manifest(tmp_path: Path):
    # Replaces the old speed != 1.0 guard, which only caught the perturbed case and blocked the
    # legitimate one. Row order IS the cache index, so ANY mismatched pairing reads plausible
    # shapes and trains on the wrong audio -- including two manifests that are both unperturbed.
    from src.slices.ExtractFeatures.FeatureCache import manifest_fingerprint

    def _manifest(name: str, uttid: str) -> str:
        path = tmp_path / name
        path.write_text(
            json.dumps({"uttid": uttid, "audio_filepath": "x", "text": "HI", "num_samples": 9000})
            + "\n",
            encoding="utf-8",
        )
        return str(path)

    built_for = _manifest("built.jsonl", "u0")
    other = _manifest("other.jsonl", "u1")
    write_feature_cache(
        str(tmp_path),
        "toy",
        [np.random.randn(6, 80).astype(np.float32)],
        manifest_fingerprint(built_for),
    )
    cache = FeatureCacheReader(str(tmp_path), "toy")
    LibriSpeechDataset(built_for, _Tok(), cache=cache)  # matching: fine
    with pytest.raises(ValueError, match="manifest"):
        LibriSpeechDataset(other, _Tok(), cache=cache)


def test_dataset_reads_perturbed_rows_from_a_matching_cache(tmp_path):
    # The old guard raised on any speed != 1.0 whenever a cache was present, which forced
    # cache-free training and a CPU-bound loader. A cache built FOR the perturbed manifest already
    # has the resampled mel baked in, so the row's speed must not be re-applied.
    import json
    import numpy as np
    from src.slices.ExtractFeatures.FeatureCache import manifest_fingerprint, write_feature_cache
    from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
    from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset

    manifest = tmp_path / "sp.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {"uttid": f"u{i}", "speed": 0.9, "audio_filepath": f"/x/{i}.flac", "text": "hi"}
            )
            for i in range(2)
        )
    )
    mels = [np.full((5, 80), float(i), dtype=np.float16) for i in range(2)]
    write_feature_cache(str(tmp_path), "sp", mels, manifest_fingerprint(str(manifest)))

    class _Tok:
        def encode(self, text: str) -> list[int]:
            return [1, 2]

    cache = FeatureCacheReader(str(tmp_path), "sp", manifest_fingerprint(str(manifest)))
    ds = LibriSpeechDataset(str(manifest), _Tok(), cache=cache)
    mel, ids = ds[1]
    assert mel.shape == (5, 80)
    assert float(mel[0, 0]) == 1.0
    assert ids == [1, 2]
