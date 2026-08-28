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


def test_speed_perturb_corrects_num_samples_in_both_directions(tmp_path):
    # sox `speed s` resamples by 1/s: 0.9 is LONGER, 1.1 is SHORTER. The sampler buckets on
    # num_samples, so getting the direction wrong blows the frame budget it exists to cap.
    from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
    from src.slices.BuildManifest.SpeedPerturbManifest_Handler import (
        build_speed_perturb_manifest,
    )

    src = tmp_path / "train.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(
                {"uttid": f"a-{i}", "audio_filepath": "x.flac", "text": "HI", "num_samples": 9000}
            )
            for i in range(40)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "train_sp2.jsonl"
    n = build_speed_perturb_manifest(SpeedPerturbManifestCommand(str(src), str(out)))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert n == 80
    clean = [r for r in rows if r["speed"] == 1.0]
    assert len(clean) == 40
    assert all(r["num_samples"] == 9000 and "_sp" not in r["uttid"] for r in clean)
    slow = next(r for r in rows if r["speed"] == 0.9)
    fast = next(r for r in rows if r["speed"] == 1.1)
    assert slow["num_samples"] == round(9000 / 0.9) > 9000
    assert fast["num_samples"] == round(9000 / 1.1) < 9000
    assert slow["uttid"].endswith("_sp0.9") and fast["uttid"].endswith("_sp1.1")


def test_speed_perturb_emits_each_utterance_once_clean_and_once_perturbed(tmp_path):
    # 2x, not 3x: disk. Each utterance keeps its original and gets ONE perturbed copy, with the
    # factor chosen per utterance so the corpus still covers both directions.
    import json
    from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
    from src.slices.BuildManifest.SpeedPerturbManifest_Handler import build_speed_perturb_manifest

    src = tmp_path / "in.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(
                {
                    "uttid": f"u{i}",
                    "audio_filepath": f"/x/{i}.flac",
                    "num_samples": 16000,
                    "text": "hi",
                }
            )
            for i in range(200)
        )
    )
    out = tmp_path / "out.jsonl"
    written = build_speed_perturb_manifest(
        SpeedPerturbManifestCommand(str(src), str(out), speeds=(0.9, 1.1), seed=1234)
    )

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == 400 and len(rows) == 400
    assert sum(1 for r in rows if r["speed"] == 1.0) == 200
    factors = {r["speed"] for r in rows if r["speed"] != 1.0}
    assert factors == {0.9, 1.1}, "both directions must appear across the corpus"
    # Perturbed rows carry a distinct uttid and a corrected length, or the sampler's frame budget
    # is wrong by up to 11 %.
    slow = next(r for r in rows if r["speed"] == 0.9)
    assert slow["uttid"].endswith("_sp0.9")
    assert slow["num_samples"] == round(16000 / 0.9)


def test_speed_perturb_assignment_is_deterministic(tmp_path):
    # Cache row order is the contract between manifest and mmap. A reshuffle between the manifest
    # build and the feature extraction would silently pair every utterance with someone else's mel.
    import json
    from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand
    from src.slices.BuildManifest.SpeedPerturbManifest_Handler import build_speed_perturb_manifest

    src = tmp_path / "in.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(
                {
                    "uttid": f"u{i}",
                    "audio_filepath": f"/x/{i}.flac",
                    "num_samples": 16000,
                    "text": "hi",
                }
            )
            for i in range(50)
        )
    )
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for out in (first, second):
        build_speed_perturb_manifest(
            SpeedPerturbManifestCommand(str(src), str(out), speeds=(0.9, 1.1), seed=1234)
        )
    assert first.read_text() == second.read_text()
