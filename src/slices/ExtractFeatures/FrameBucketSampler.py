# src/slices/ExtractFeatures/FrameBucketSampler.py
import json

from torch.utils.data import Sampler

from src.shared_kernel.Config_Adapter import get_config


class FrameBucketSampler(Sampler):
    """Groups utterances of similar length so each batch fills a frame budget,
    keeping padding low and VRAM near-constant across batches."""

    def __init__(self, manifest: str, max_frames_per_batch: int) -> None:
        hop_length = get_config().audio.hop_length
        rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._frames = [r["num_samples"] // hop_length for r in rows]
        self._order = sorted(range(len(rows)), key=lambda i: self._frames[i])
        self._max_frames = max_frames_per_batch

    def __iter__(self):
        batch: list[int] = []
        budget = 0
        for idx in self._order:
            if batch and budget + self._frames[idx] > self._max_frames:
                yield batch
                batch, budget = [], 0
            batch.append(idx)
            budget += self._frames[idx]
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(self._frames)
        return max(1, total // self._max_frames)
