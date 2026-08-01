import glob
import torch
from src.shared_kernel.AudioIO_Adapter import load_audio, speed_perturb
from src.shared_kernel.Config_Adapter import get_config


def test_load_audio_returns_mono_16k():
    path = glob.glob("data/Val/dev-clean/**/*.flac", recursive=True)[0]
    wave = load_audio(path)
    assert wave.dtype == torch.float32
    assert wave.ndim == 1
    assert wave.numel() > get_config().audio.sample_rate // 10  # at least 0.1 s of audio


def test_speed_perturb_identity_is_noop():
    wave = torch.randn(16000)
    assert torch.equal(speed_perturb(wave, 1.0), wave)


def test_speed_perturb_changes_length_by_inverse_speed():
    # sox-style: length scales by 1/speed, so 0.9 lengthens ~11% and 1.1 shortens ~9%. The manifest
    # generator uses round(num_samples / speed); both must agree so the frame budget stays accurate.
    n = 16000
    wave = torch.randn(n)
    slow = speed_perturb(wave, 0.9)
    fast = speed_perturb(wave, 1.1)
    assert abs(slow.numel() - round(n / 0.9)) <= 2
    assert abs(fast.numel() - round(n / 1.1)) <= 2
