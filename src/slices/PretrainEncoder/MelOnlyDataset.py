# src/slices/PretrainEncoder/MelOnlyDataset.py
# BEST-RQ ignores transcripts, so pretraining reads only cached log-mel (SP1 fp16 mmap cache) — the
# lightest possible item path, keeping the pretrain loop GPU-bound (SP4).
import json

import torch
from torch.utils.data import Dataset

from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader


class MelOnlyDataset(Dataset):
    def __init__(self, manifest: str, cache: FeatureCacheReader) -> None:
        self._rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._cache = cache

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._cache[index]  # [T, 80] float32
