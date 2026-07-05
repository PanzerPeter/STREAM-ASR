# src/slices/ComputeCmvn/ComputeCmvn_Handler.py
import json
import os

import torch

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ComputeCmvn.ComputeCmvn_Command import ComputeCmvnCommand


def compute_cmvn(cmd: ComputeCmvnCommand) -> dict:
    if not os.path.isfile(cmd.manifest):
        raise FileNotFoundError(cmd.manifest)

    audio = get_config().audio
    rows = [json.loads(line) for line in open(cmd.manifest, encoding="utf-8")]
    if cmd.max_utts is not None:
        rows = rows[: cmd.max_utts]

    # Streaming accumulation over the mel bins keeps memory flat regardless of corpus size.
    total = torch.zeros(audio.n_mels, dtype=torch.float64)
    total_sq = torch.zeros(audio.n_mels, dtype=torch.float64)
    count = 0

    for row in rows:
        mel = compute_log_mel(load_audio(row["audio_filepath"])).double()  # [T, 80]
        total += mel.sum(dim=0)
        total_sq += (mel * mel).sum(dim=0)
        count += mel.shape[0]

    mean = total / count
    var = (total_sq / count) - mean * mean
    std = var.clamp_min(audio.cmvn_eps).sqrt()

    stats = {"mean": mean.float(), "std": std.float()}
    os.makedirs(os.path.dirname(cmd.cmvn_out) or ".", exist_ok=True)
    torch.save(stats, cmd.cmvn_out)
    return stats
