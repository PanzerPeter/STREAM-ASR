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
    write_feature_cache(str(tmp_path), "toy", mels, "fp-toy")
    reader = FeatureCacheReader(str(tmp_path), "toy")
    assert len(reader) == 2
    got = reader[0]
    assert isinstance(got, torch.Tensor) and got.shape == (5, n_mels)
    assert torch.allclose(got, torch.from_numpy(mels[0]), atol=1e-2)  # fp16 tolerance


def test_cache_header_mismatch_raises(tmp_path, monkeypatch):
    n_mels = get_config().audio.n_mels
    write_feature_cache(str(tmp_path), "toy", [np.zeros((2, n_mels), np.float32)], "fp-toy")
    import json

    hp = tmp_path / "toy.header.json"
    h = json.loads(hp.read_text())
    h["n_mels"] = n_mels + 1
    hp.write_text(json.dumps(h))
    with pytest.raises(ValueError):
        FeatureCacheReader(str(tmp_path), "toy")


def test_cache_rejects_a_manifest_it_was_not_built_for(tmp_path):
    # THE failure mode speed perturbation introduces: a clean-mel cache paired with a perturbed
    # manifest reads the right number of rows and the right shapes, and trains on the wrong audio.
    # Nothing else in the pipeline would notice.
    import json
    import numpy as np
    import pytest
    from src.slices.ExtractFeatures.FeatureCache import (
        FeatureCacheReader,
        manifest_fingerprint,
        write_feature_cache,
    )

    def _manifest(path, speed):
        path.write_text(
            "\n".join(
                json.dumps({"uttid": f"u{i}", "speed": speed, "audio_filepath": f"/x/{i}.flac"})
                for i in range(3)
            )
        )
        return str(path)

    clean = _manifest(tmp_path / "clean.jsonl", 1.0)
    perturbed = _manifest(tmp_path / "sp.jsonl", 0.9)
    n_mels = 80
    mels = [np.zeros((4, n_mels), dtype=np.float16) for _ in range(3)]
    write_feature_cache(str(tmp_path), "train", mels, manifest_fingerprint(clean))

    FeatureCacheReader(str(tmp_path), "train", manifest_fingerprint(clean))  # matching: fine
    with pytest.raises(ValueError, match="manifest"):
        FeatureCacheReader(str(tmp_path), "train", manifest_fingerprint(perturbed))
