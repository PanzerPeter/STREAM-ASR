import json

from torch.utils.data import Dataset

from src.shared_kernel.AudioIO_Adapter import load_audio, speed_perturb
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader, manifest_fingerprint


class LibriSpeechDataset(Dataset):
    """Yields (log-mel, token_ids). With a feature cache, __getitem__ is an mmap slice, so the epoch
    loop stays GPU-bound. Spectral augmentation happens later, as a GPU batch op in the trainer
    (`SpecAugmentBatch`). Speed perturbation is an audio-domain augmentation keyed off the row's
    `speed` field (set by SpeedPerturbManifest); a cache is built PER MANIFEST and already carries
    it, so the cached path must not re-apply it. The cache is bound to its manifest by a
    fingerprint, checked here at construction."""

    def __init__(
        self,
        manifest: str,
        tokenizer,
        cache: FeatureCacheReader | None = None,
    ) -> None:
        self._rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._tokenizer = tokenizer
        self._cache = cache
        # Row order IS the cache index, so a cache built for another manifest reads plausible
        # shapes and trains on the wrong audio. Fail at construction, not never.
        if cache is not None and cache.fingerprint is not None:
            expected = manifest_fingerprint(manifest)
            if cache.fingerprint != expected:
                raise ValueError(
                    f"feature cache was built for a different manifest than {manifest} "
                    f"(cache {cache.fingerprint}, manifest {expected})"
                )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        if self._cache is not None:
            mel = self._cache[index]
        else:
            speed = row.get("speed", 1.0)
            mel = compute_log_mel(speed_perturb(load_audio(row["audio_filepath"]), speed))
        return mel, self._tokenizer.encode(row["text"])
