# src/slices/ExtractFeatures/FeatureCache.py
# fp16 log-mel cache: one flat memmap per split streamed via mmap, so the training epoch loop is
# GPU-bound (no per-epoch FLAC decode / FFT). A header records the front-end params the cache was
# built with; a mismatch against config fails loudly rather than feeding stale features silently.
import json
import os
from typing import Iterable

import numpy as np
import torch

from src.shared_kernel.Config_Adapter import get_config

_HEADER_KEYS = ("sample_rate", "n_mels", "n_fft", "win_length", "hop_length")


def _header_from_config() -> dict[str, int]:
    a = get_config().audio
    return {k: getattr(a, k) for k in _HEADER_KEYS}


def write_feature_cache(cache_dir: str, split: str, mels: Iterable[np.ndarray]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    n_mels = get_config().audio.n_mels
    flat_path = os.path.join(cache_dir, f"{split}.f16")
    index: list[tuple[int, int]] = []
    offset = 0
    with open(flat_path, "wb") as sink:
        for mel in mels:
            arr = np.ascontiguousarray(mel, dtype=np.float16)
            if arr.ndim != 2 or arr.shape[1] != n_mels:
                raise ValueError(f"mel shape {arr.shape} != [T, {n_mels}]")
            sink.write(arr.tobytes())
            index.append((offset, arr.shape[0]))
            offset += arr.shape[0]
    np.save(os.path.join(cache_dir, f"{split}.index.npy"), np.asarray(index, dtype=np.int64))
    header: dict[str, object] = dict(_header_from_config())
    header.update({"dtype": "float16", "num_utts": len(index), "total_frames": offset})
    with open(os.path.join(cache_dir, f"{split}.header.json"), "w", encoding="utf-8") as f:
        json.dump(header, f)


class FeatureCacheReader:
    def __init__(self, cache_dir: str, split: str) -> None:
        with open(os.path.join(cache_dir, f"{split}.header.json"), encoding="utf-8") as f:
            header = json.load(f)
        expected = _header_from_config()
        for k in _HEADER_KEYS:
            if header.get(k) != expected[k]:
                raise ValueError(
                    f"feature cache {split}: header {k}={header.get(k)} != config {expected[k]}"
                )
        self._index: np.ndarray = np.load(os.path.join(cache_dir, f"{split}.index.npy"))
        self._mel: np.memmap = np.memmap(
            os.path.join(cache_dir, f"{split}.f16"),
            dtype=np.float16,
            mode="r",
            shape=(int(header["total_frames"]), int(header["n_mels"])),
        )

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, i: int) -> torch.Tensor:
        offset, num_frames = int(self._index[i, 0]), int(self._index[i, 1])
        chunk = np.asarray(self._mel[offset : offset + num_frames], dtype=np.float32)
        return torch.from_numpy(chunk)
