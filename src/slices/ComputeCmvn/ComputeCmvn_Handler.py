import json
import multiprocessing as mp
import os
import random
from typing import Any

import numpy as np
import torch

from src.shared_kernel.AudioIO_Adapter import load_row_audio
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand


def _init_worker() -> None:
    # One intra-op thread per worker: the parallelism that pays is across utterances, and a
    # machine-sized torch thread pool per process only spins against the others.
    torch.set_num_threads(1)


def _moments(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    mel = compute_log_mel(load_row_audio(row)).double()  # [T, n_mels]
    return mel.sum(dim=0).numpy(), (mel * mel).sum(dim=0).numpy(), int(mel.shape[0])


def compute_cmvn(cmd: ComputeCmvnCommand) -> dict:
    for manifest in cmd.manifests:
        if not os.path.isfile(manifest):
            raise FileNotFoundError(manifest)

    audio = get_config().audio
    rows = [json.loads(line) for m in cmd.manifests for line in open(m, encoding="utf-8")]
    if cmd.sample_frac is not None:
        k = max(1, int(len(rows) * cmd.sample_frac))
        rows = random.Random(cmd.seed).sample(rows, k)
    if cmd.max_utts is not None:
        rows = rows[: cmd.max_utts]

    # Accumulated in float64 over the mel bins, so memory stays flat regardless of corpus size.
    # imap is ordered, which keeps the float sums (and so the stats) identical for a given seed.
    total = np.zeros(audio.n_mels, dtype=np.float64)
    total_sq = np.zeros(audio.n_mels, dtype=np.float64)
    count = 0
    with mp.get_context("spawn").Pool(cmd.num_workers, initializer=_init_worker) as pool:
        for s, sq, n in pool.imap(_moments, rows, chunksize=32):
            total += s
            total_sq += sq
            count += n

    mean = torch.from_numpy(total / count)
    var = torch.from_numpy(total_sq / count) - mean * mean
    std = var.clamp_min(audio.cmvn_eps).sqrt()

    stats = {"mean": mean.float(), "std": std.float()}
    os.makedirs(os.path.dirname(cmd.cmvn_out) or ".", exist_ok=True)
    torch.save(stats, cmd.cmvn_out)
    return stats
