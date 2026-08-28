# One-time pass: decode + log-mel every utterance and write the fp16 memmap cache. Decoding is the
# cost, so it runs in a process pool; imap preserves manifest row order (the collator/sampler index
# the cache by row), and the main process streams results straight into write_feature_cache.
import json
import multiprocessing as mp
import time
from typing import Iterator

import numpy as np
import torch

from src.shared_kernel.AudioIO_Adapter import load_audio, speed_perturb
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Logging_Adapter import configure_logging
from src.slices.ExtractFeatures.FeatureCache import (
    cached_utt_count,
    manifest_fingerprint,
    write_feature_cache,
)
from src.slices.ExtractFeatures.PrecomputeFeatures_Command import PrecomputeFeaturesCommand

# One line every this many utts: decode is minutes-to-hours per split, so a silent stream reads as a
# hang. Frequent enough to show live throughput, sparse enough not to spam a 280k-utt train pass.
_LOG_EVERY = 2000

# "spawn", not the platform-default "fork": torch/torchaudio initialize internal thread pools on
# import, and forking a multi-threaded parent risks a worker deadlocking on a mutex that was held by
# a non-forked thread at fork time (classic PyTorch + fork hazard). Scoped to a local context so it
# doesn't change the start method process-wide (e.g. DataLoader workers elsewhere).
_CTX = mp.get_context("spawn")


def _init_worker() -> None:
    # One intra-op thread per worker. torch otherwise gives EVERY pool process a thread pool sized
    # to the whole machine, so N workers each fan a 6 ms STFT out over ~20 threads that spin-wait
    # far longer than the op takes: measured 8 workers held ~18 cores busy to produce 26 utt/s,
    # while a single thread costs 14 ms/utt (decode 6 + log-mel 8) = 71 utt/s. The parallelism that
    # pays here is across utterances, which the pool already provides.
    torch.set_num_threads(1)


def _mel_for(job: tuple[str, float]) -> np.ndarray:
    # The row's speed factor is applied HERE, so the perturbation is baked into the cache and the
    # training loader stays an mmap slice. speed == 1.0 makes speed_perturb a no-op.
    path, speed = job
    return compute_log_mel(speed_perturb(load_audio(path), speed)).numpy().astype(np.float16)


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
    log = configure_logging()
    fingerprint = manifest_fingerprint(cmd.manifest)
    done = cached_utt_count(cmd.cache_dir, cmd.split, fingerprint)
    if done is not None:
        log.info(f"{cmd.split}: {done} utts already cached for this manifest, skipping")
        return done

    rows = [json.loads(line) for line in open(cmd.manifest, encoding="utf-8")]
    jobs = [(row["audio_filepath"], row.get("speed", 1.0)) for row in rows]

    def _mels() -> Iterator[np.ndarray]:
        with _CTX.Pool(cmd.num_workers, initializer=_init_worker) as pool:
            yield from pool.imap(_mel_for, jobs, chunksize=64)  # ordered

    write_feature_cache(
        cmd.cache_dir, cmd.split, _log_progress(_mels(), cmd.split, len(jobs)), fingerprint
    )
    return len(jobs)
