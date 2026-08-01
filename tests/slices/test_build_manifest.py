import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand
from src.slices.BuildManifest.BuildManifest_Handler import build_manifest


def _fixture_split(root: Path) -> None:
    chapter = root / "spk" / "chap"
    chapter.mkdir(parents=True)
    sr = 16000
    for uid, n in [("b-2", 16000), ("a-1", 8000)]:  # deliberately out of sorted order
        sf.write(chapter / f"{uid}.flac", np.zeros(n, dtype=np.float32), sr)
    (chapter / "chap.trans.txt").write_text("b-2 HELLO WORLD\na-1 FOO BAR\n", encoding="utf-8")


def test_build_manifest_parallel_sorted(tmp_path):
    _fixture_split(tmp_path)
    out = tmp_path / "m.jsonl"
    rows = build_manifest(BuildManifestCommand(str(tmp_path), str(out)))
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows == 2
    assert [r["uttid"] for r in lines] == ["a-1", "b-2"]  # sorted by uttid
    assert lines[0]["num_samples"] == 8000
    assert lines[1]["num_samples"] == 16000


def test_speed_perturb_manifest_triples_rows_and_corrects_samples(tmp_path):
    from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
    from src.slices.BuildManifest.SpeedPerturbManifest_Handler import (
        build_speed_perturb_manifest,
    )

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"uttid": "a-1", "audio_filepath": "x.flac", "text": "HI", "num_samples": 9000})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "train_sp.jsonl"
    n = build_speed_perturb_manifest(SpeedPerturbManifestCommand(str(src), str(out)))
    rows = {r["speed"]: r for r in (json.loads(x) for x in out.read_text().splitlines())}
    assert n == 3 and set(rows) == {0.9, 1.0, 1.1}
    assert rows[1.0]["uttid"] == "a-1" and rows[1.0]["num_samples"] == 9000  # original untouched
    assert rows[0.9]["uttid"] == "a-1_sp0.9" and rows[0.9]["num_samples"] == round(9000 / 0.9)
    assert rows[1.1]["num_samples"] == round(9000 / 1.1)  # faster -> fewer samples
