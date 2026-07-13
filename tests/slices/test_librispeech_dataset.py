import json
from pathlib import Path

import numpy as np
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
    ds = LibriSpeechDataset(
        str(manifest), _Tok(), train=True, cache=FeatureCacheReader(str(tmp_path), "toy")
    )
    mel, ids = ds[0]
    assert mel.shape == (6, 80)
    assert torch.allclose(mel, torch.from_numpy(mel0), atol=1e-2)  # cache read, no augmentation
    assert ids == [ord("A") % 7, ord("B") % 7]
