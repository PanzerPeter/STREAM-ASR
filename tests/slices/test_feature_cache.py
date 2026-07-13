import numpy as np
import pytest
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader, write_feature_cache


def test_cache_roundtrip(tmp_path):
    n_mels = get_config().audio.n_mels
    mels = [
        np.random.randn(5, n_mels).astype(np.float32),
        np.random.randn(3, n_mels).astype(np.float32),
    ]
    write_feature_cache(str(tmp_path), "toy", mels)
    reader = FeatureCacheReader(str(tmp_path), "toy")
    assert len(reader) == 2
    got = reader[0]
    assert isinstance(got, torch.Tensor) and got.shape == (5, n_mels)
    assert torch.allclose(got, torch.from_numpy(mels[0]), atol=1e-2)  # fp16 tolerance


def test_cache_header_mismatch_raises(tmp_path, monkeypatch):
    n_mels = get_config().audio.n_mels
    write_feature_cache(str(tmp_path), "toy", [np.zeros((2, n_mels), np.float32)])
    import json

    hp = tmp_path / "toy.header.json"
    h = json.loads(hp.read_text())
    h["n_mels"] = n_mels + 1
    hp.write_text(json.dumps(h))
    with pytest.raises(ValueError):
        FeatureCacheReader(str(tmp_path), "toy")
