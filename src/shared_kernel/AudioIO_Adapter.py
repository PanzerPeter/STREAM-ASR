# Audio load/resample adapter. torchaudio 2.11 removed its native decode backends (it delegates to
# TorchCodec/FFmpeg), so every decode goes through soundfile, which reads FLAC via libsndfile with
# no extra system deps; torchaudio is kept only for its pure-tensor resample kernel.
import io
import json
from fractions import Fraction
from typing import Any, BinaryIO

import soundfile as sf
import torch
import torchaudio

from src.shared_kernel.Config_Adapter import get_config

# limit_denominator keeps the resample ratio a small coprime integer pair (9/10, 11/10, ...). Raw
# sample rates (e.g. 14400 -> 16000) are the coprime-resample footgun: torchaudio then builds a
# needlessly huge polyphase kernel. 20 covers every standard speed factor to 3 decimals.
_SPEED_RATIO_MAX_DENOM = 20


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


def speed_perturb(wave: torch.Tensor, speed: float) -> torch.Tensor:
    # sox-style `speed s`: resample by 1/s and reinterpret at the original rate, changing tempo AND
    # pitch. Output length ~= len / s. speed == 1.0 is a no-op. Fraction(1/s) num/denominator
    # ARE the resample orig/new freqs, so a 0.9 factor resamples 9 -> 10 (length x10/9 = len/0.9).
    if speed == 1.0:
        return wave
    ratio = Fraction(1.0 / speed).limit_denominator(_SPEED_RATIO_MAX_DENOM)
    return torchaudio.functional.resample(wave, ratio.denominator, ratio.numerator)


def load_audio(path: str) -> torch.Tensor:
    return _decode(path)


def load_audio_bytes(raw: bytes) -> torch.Tensor:
    # Decode an uploaded audio file held in memory (WAV/FLAC/OGG via libsndfile, no FFmpeg needed).
    return _decode(io.BytesIO(raw))


def load_manifest(path: str) -> list[dict[str, Any]]:
    # A LibriSpeech manifest is one JSON object per line (uttid, audio_filepath, text, num_samples).
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
