# Build the fp16 log-mel cache for every split the trainer reads (heavy; one-time).
#
# `train_sp2` is the paired 2x speed-perturbed LibriSpeech manifest (~105 GB): its mels are
# resampled at extraction and baked into the cache. `cv-train` (~100 GB) bakes in its silence trim
# the same way. The clean `train` manifest and the full cv-dev / cv-test sets are never read from a
# cache (the trainer reads train_sp2 and cv-dev2k; Evaluate decodes audio), so they get none. Check
# free disk before running. Each cache is bound to the manifest it was built from by a fingerprint
# in its header, so pairing the wrong two is a load-time error rather than silent mistraining.
#
# Re-runnable: a split whose cache is already complete for its manifest is skipped, so an
# interrupted pass is restarted with the same command and only redoes the split it died in
# (the granularity is a whole split -- there is no mid-split resume). Delete a split's
# `<split>.header.json` to force it to be rebuilt.
from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand
from src.slices.ExtractFeatures.PrecomputeFeatures_Handler import precompute_features

SPLITS = [
    ("data/manifests/train_sp2.jsonl", "train_sp2"),
    ("data/manifests/cv-train.jsonl", "cv-train"),
    ("data/manifests/dev.jsonl", "dev"),
    ("data/manifests/dev-other.jsonl", "dev-other"),
    ("data/manifests/cv-dev2k.jsonl", "cv-dev2k"),
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
