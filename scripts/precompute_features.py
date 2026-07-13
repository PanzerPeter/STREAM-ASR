# scripts/precompute_features.py — build the fp16 log-mel cache for every split (heavy; one-time)
from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand
from src.slices.ExtractFeatures.PrecomputeFeatures_Handler import precompute_features

SPLITS = [
    ("data/manifests/train.jsonl", "train"),
    ("data/manifests/dev.jsonl", "dev"),
    ("data/manifests/dev-other.jsonl", "dev-other"),
    ("data/manifests/test.jsonl", "test"),
    ("data/manifests/test-other.jsonl", "test-other"),
]


def main() -> None:
    cache_dir = get_config().features.cache_dir
    for manifest, split in SPLITS:
        n = precompute_features(PrecomputeFeaturesCommand(manifest, split, cache_dir))
        print(f"{split}: cached {n} utts -> {cache_dir}")


if __name__ == "__main__":
    main()
