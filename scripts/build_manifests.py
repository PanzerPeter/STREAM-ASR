# scripts/build_manifests.py — build all 960h + eval manifests
from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand
from src.slices.BuildManifest.BuildManifest_Handler import build_manifest

SPLITS = [
    ("data/Train", "data/manifests/train.jsonl"),  # clean-100 + clean-360 + other-500
    ("data/Val/dev-clean", "data/manifests/dev.jsonl"),
    ("data/Val/dev-other", "data/manifests/dev-other.jsonl"),
    ("data/Test/test-clean", "data/manifests/test.jsonl"),
    ("data/Test/test-other", "data/manifests/test-other.jsonl"),
]


def main() -> None:
    for split_dir, out in SPLITS:
        rows = build_manifest(BuildManifestCommand(split_dir, out))
        print(f"{out}: {rows} utts")


if __name__ == "__main__":
    main()
