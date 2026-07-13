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
