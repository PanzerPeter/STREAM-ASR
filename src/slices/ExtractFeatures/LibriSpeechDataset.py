import json

from torch.utils.data import Dataset

from src.shared_kernel.AudioIO_Adapter import load_audio, speed_perturb
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader


class LibriSpeechDataset(Dataset):
    """Yields (log-mel, token_ids). With a feature cache, __getitem__ is an mmap slice — the epoch
    loop stays GPU-bound. Spectral augmentation happens later, as a GPU batch op in the trainer
    (`SpecAugmentBatch`). The one audio-domain augmentation applied here is 3-way speed
    perturbation, keyed off the row's `speed` field (set by SpeedPerturbManifest); it resamples the
    waveform, so it is incompatible with the precomputed clean-mel cache."""

    def __init__(
        self,
        manifest: str,
        tokenizer,
        cache: FeatureCacheReader | None = None,
    ) -> None:
        self._rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._tokenizer = tokenizer
        self._cache = cache

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        speed = row.get("speed", 1.0)
        if self._cache is not None:
            if speed != 1.0:
                raise ValueError(
                    "speed-perturbed rows have no cached features; train cache-free on the "
                    "*_sp.jsonl manifest so audio is resampled per row at load time."
                )
            mel = self._cache[index]
        else:
            mel = compute_log_mel(speed_perturb(load_audio(row["audio_filepath"]), speed))
        return mel, self._tokenizer.encode(row["text"])
