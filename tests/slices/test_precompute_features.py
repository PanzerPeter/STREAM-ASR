import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand
from src.slices.ExtractFeatures.PrecomputeFeatures_Handler import precompute_features


def test_precompute_matches_online(tmp_path: Path):
    manifest = tmp_path / "m.jsonl"
    paths = []
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(3):
            p = tmp_path / f"u{i}.flac"
            sf.write(p, np.random.RandomState(i).randn(4000 + i * 800).astype(np.float32), 16000)
            paths.append(str(p))
            f.write(
                json.dumps({"audio_filepath": str(p), "text": "X", "num_samples": 4000 + i * 800})
                + "\n"
            )
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
