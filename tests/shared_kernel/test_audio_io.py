import glob
import torch
from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Config_Adapter import get_config


def test_load_audio_returns_mono_16k():
    path = glob.glob("data/Val/dev-clean/**/*.flac", recursive=True)[0]
    wave = load_audio(path)
    assert wave.dtype == torch.float32
    assert wave.ndim == 1
    assert wave.numel() > get_config().audio.sample_rate // 10  # at least 0.1 s of audio
