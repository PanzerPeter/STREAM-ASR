# src/slices/ExtractFeatures/LibriSpeechDataset.py
import json

from torch.utils.data import Dataset

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader


class LibriSpeechDataset(Dataset):
    """Yields (clean log-mel, token_ids). With a feature cache, __getitem__ is an mmap slice — the
    epoch loop stays GPU-bound. Augmentation is not applied here: SpecAugment runs as a GPU batch
    op in the trainer (see SpecAugmentBatch); speed-perturb was dropped (see SP1 design)."""

    def __init__(
        self,
        manifest: str,
        tokenizer,
        train: bool,
        cache: FeatureCacheReader | None = None,
    ) -> None:
        self._rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._tokenizer = tokenizer
        self._cache = cache

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        if self._cache is not None:
            mel = self._cache[index]
        else:
            mel = compute_log_mel(load_audio(row["audio_filepath"]))
        return mel, self._tokenizer.encode(row["text"])
