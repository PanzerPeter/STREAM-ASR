# Build the fp16 log-mel cache for every split (heavy; one-time).
#
# `train_sp2` is the paired 2x speed-perturbed train manifest: its mels are resampled at extraction
# and baked into the cache, so it costs ~53 GB ON TOP of the clean `train` cache. Check free disk
# before running. Each cache is bound to the manifest it was built from by a fingerprint in its
# header, so pairing the wrong two is a load-time error rather than silent mistraining.
#
# Re-runnable: a split whose cache is already complete for its manifest is skipped, so an
# interrupted pass is restarted with the same command and only redoes the split it died in
# (the granularity is a whole split -- there is no mid-split resume). Delete a split's
# `<split>.header.json` to force it to be rebuilt.
from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand
from src.slices.ExtractFeatures.PrecomputeFeatures_Handler import precompute_features

SPLITS = [
    ("data/manifests/train.jsonl", "train"),
    ("data/manifests/train_sp2.jsonl", "train_sp2"),
    ("data/manifests/dev.jsonl", "dev"),
    ("data/manifests/dev-other.jsonl", "dev-other"),
    ("data/manifests/test.jsonl", "test"),
    ("data/manifests/test-other.jsonl", "test-other"),
]


def main() -> None:
    cache_dir = get_config().features.cache_dir
    for manifest, split in SPLITS:
        n = precompute_features(PrecomputeFeaturesCommand(manifest, split, cache_dir))
        print(f"{split}: {n} utts cached in {cache_dir}")


if __name__ == "__main__":
    main()
