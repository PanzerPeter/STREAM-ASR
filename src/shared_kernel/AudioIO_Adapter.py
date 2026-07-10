# src/shared_kernel/AudioIO_Adapter.py
import io
import json
from typing import Any, BinaryIO

import soundfile as sf
import torch
import torchaudio

from src.shared_kernel.Config_Adapter import get_config

# torchaudio 2.11 removed its native decode backends (delegates to TorchCodec/FFmpeg).
# soundfile reads FLAC via libsndfile with no extra system deps, so decode goes through it;
# torchaudio is kept only for the pure-tensor resample kernel.


def _decode(src: str | BinaryIO) -> torch.Tensor:
    # soundfile.read accepts a path or any file-like (the demo server passes uploaded bytes), so the
    # downmix + resample-to-config-rate path is shared by both the file and in-memory entry points.
    sample_rate = get_config().audio.sample_rate
    data, sr = sf.read(src, dtype="float32", always_2d=True)  # [num_frames, channels]
    wave = torch.from_numpy(data).transpose(0, 1)  # -> [channels, num_frames]

    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)  # downmix; LibriSpeech is mono but guard anyway

    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)

    return wave.squeeze(0).to(torch.float32)


def load_audio(path: str) -> torch.Tensor:
    return _decode(path)


def load_audio_bytes(raw: bytes) -> torch.Tensor:
    # Decode an uploaded audio file held in memory (WAV/FLAC/OGG via libsndfile — no FFmpeg needed).
    return _decode(io.BytesIO(raw))


def load_manifest(path: str) -> list[dict[str, Any]]:
    # A LibriSpeech manifest is one JSON object per line (uttid, audio_filepath, text, num_samples).
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
