# src/shared_kernel/AudioIO_Adapter.py
import soundfile as sf
import torch
import torchaudio

from src.shared_kernel.Config_Adapter import get_config

# torchaudio 2.11 removed its native decode backends (delegates to TorchCodec/FFmpeg).
# soundfile reads FLAC via libsndfile with no extra system deps, so decode goes through it;
# torchaudio is kept only for the pure-tensor resample kernel.


def load_audio(path: str) -> torch.Tensor:
    sample_rate = get_config().audio.sample_rate
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [num_frames, channels]
    wave = torch.from_numpy(data).transpose(0, 1)  # -> [channels, num_frames]

    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)  # downmix; LibriSpeech is mono but guard anyway

    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)

    return wave.squeeze(0).to(torch.float32)
