# Fp16 log-mel cache: one flat memmap per split streamed via mmap, so the training epoch loop is
# GPU-bound (no per-epoch FLAC decode / FFT). A header records the front-end params the cache was
# built with; a mismatch against config fails loudly rather than feeding stale features silently.
import hashlib
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


def manifest_fingerprint(manifest_path: str) -> str:
    """Content hash over the (uttid, speed) sequence a cache was built from.

    The cache is a flat memmap indexed by manifest ROW ORDER, so the manifest is not metadata --
    it is half the data structure. Pairing a clean-mel cache with a speed-perturbed manifest reads
    the right row count and the right shapes and trains on audio that belongs to a different
    utterance, which no downstream check would catch. Hashing the identity of every row makes that
    a load-time error instead of a silent one.
    """
    digest = hashlib.sha1()
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            digest.update(f"{row['uttid']}:{row.get('speed', 1.0)}\n".encode())
    return digest.hexdigest()


def cached_utt_count(cache_dir: str, split: str, fingerprint: str) -> int | None:
    """Utterance count of a cache that is COMPLETE and built from this manifest, else None.

    `write_feature_cache` streams the flat file and writes the index and header only once the last
    utterance has landed, so a header whose bookkeeping matches the bytes on disk is the completion
    marker: a pass killed mid-split leaves a short `.f16` under a stale or absent header. Building
    the train cache is hours, so `scripts/precompute_features.py` checks this per split and a
    restart redoes only the split that was interrupted. Delete the header to force a rebuild.
    """
    try:
        with open(os.path.join(cache_dir, f"{split}.header.json"), encoding="utf-8") as f:
            header = json.load(f)
        index: np.ndarray = np.load(os.path.join(cache_dir, f"{split}.index.npy"))
        flat_bytes = os.path.getsize(os.path.join(cache_dir, f"{split}.f16"))
    except (OSError, ValueError, EOFError):
        return None
    num_utts, total_frames = header.get("num_utts"), header.get("total_frames")
    if not isinstance(num_utts, int) or not isinstance(total_frames, int):
        return None
    if header.get("manifest_fingerprint") != fingerprint or header.get("dtype") != "float16":
        return None
    if any(header.get(k) != v for k, v in _header_from_config().items()):
        return None
    frame_bytes = get_config().audio.n_mels * np.dtype(np.float16).itemsize
    if index.shape != (num_utts, 2) or flat_bytes != total_frames * frame_bytes:
        return None
    return num_utts


def write_feature_cache(
    cache_dir: str, split: str, mels: Iterable[np.ndarray], fingerprint: str
) -> None:
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
    header.update(
        {
            "dtype": "float16",
            "num_utts": len(index),
            "total_frames": offset,
            "manifest_fingerprint": fingerprint,
        }
    )
    with open(os.path.join(cache_dir, f"{split}.header.json"), "w", encoding="utf-8") as f:
        json.dump(header, f)


class FeatureCacheReader:
    def __init__(self, cache_dir: str, split: str, fingerprint: str | None = None) -> None:
        with open(os.path.join(cache_dir, f"{split}.header.json"), encoding="utf-8") as f:
            header = json.load(f)
        expected = _header_from_config()
        for k in _HEADER_KEYS:
            if header.get(k) != expected[k]:
                raise ValueError(
                    f"feature cache {split}: header {k}={header.get(k)} != config {expected[k]}"
                )
        self.fingerprint = header.get("manifest_fingerprint")
        # None skips the check, for callers that built the cache inline and already know the
        # pairing. A supplied fingerprint that disagrees is fatal: see manifest_fingerprint.
        if fingerprint is not None and self.fingerprint != fingerprint:
            raise ValueError(
                f"feature cache {split}: built for a different manifest "
                f"(cache {self.fingerprint}, requested {fingerprint})"
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
