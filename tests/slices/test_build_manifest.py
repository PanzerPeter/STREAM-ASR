import glob
import json
from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand
from src.slices.BuildManifest.BuildManifest_Handler import build_manifest


def test_manifest_row_count_matches_flac_count(tmp_path):
    split = "data/Val/dev-clean"
    out = tmp_path / "dev.jsonl"
    n = build_manifest(BuildManifestCommand(split_dir=split, manifest_out=str(out)))

    flac_count = len(glob.glob(f"{split}/**/*.flac", recursive=True))
    assert n == flac_count

    first = json.loads(out.read_text().splitlines()[0])
    assert set(first) == {"uttid", "audio_filepath", "text", "num_samples"}
    assert first["text"] == first["text"].upper()
    assert first["num_samples"] > 0
