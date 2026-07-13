# src/slices/ExtractFeatures/PrecomputeFeatures_Handler.py
# One-time pass: decode + log-mel every utterance and write the fp16 memmap cache. Decoding is the
# cost, so it runs in a process pool; imap preserves manifest row order (the collator/sampler index
# the cache by row), and the main process streams results straight into write_feature_cache.
import json
import multiprocessing as mp
import time
from typing import Iterator

import numpy as np

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Logging_Adapter import configure_logging
from src.slices.ExtractFeatures.FeatureCache import write_feature_cache
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand

# One line every this many utts: decode is minutes-to-hours per split, so a silent stream reads as a
# hang. Frequent enough to show live throughput, sparse enough not to spam a 280k-utt train pass.
_LOG_EVERY = 2000

# "spawn", not the platform-default "fork": torch/torchaudio initialize internal thread pools on
# import, and forking a multi-threaded parent risks a worker deadlocking on a mutex that was held by
# a non-forked thread at fork time (classic PyTorch + fork hazard). Scoped to a local context so it
# doesn't change the start method process-wide (e.g. DataLoader workers elsewhere).
_CTX = mp.get_context("spawn")


def _mel_for(path: str) -> np.ndarray:
    return compute_log_mel(load_audio(path)).numpy().astype(np.float16)


def _log_progress(mels: Iterator[np.ndarray], split: str, total: int) -> Iterator[np.ndarray]:
    log = configure_logging()
    start = time.monotonic()
    for done, mel in enumerate(mels, 1):
        if done % _LOG_EVERY == 0 or done == total:
            rate = done / (time.monotonic() - start)  # utts/s, averaged over the whole pass
            eta_min = (total - done) / rate / 60 if rate else 0.0
            log.info(
                f"{split}: {done}/{total} ({done / total:.0%}) {rate:.0f} utt/s ETA {eta_min:.0f}m"
            )
        yield mel


def precompute_features(cmd: PrecomputeFeaturesCommand) -> int:
    paths = [json.loads(line)["audio_filepath"] for line in open(cmd.manifest, encoding="utf-8")]

    def _mels() -> Iterator[np.ndarray]:
        with _CTX.Pool(cmd.num_workers) as pool:
            yield from pool.imap(_mel_for, paths, chunksize=64)  # ordered

    write_feature_cache(cmd.cache_dir, cmd.split, _log_progress(_mels(), cmd.split, len(paths)))
    return len(paths)
