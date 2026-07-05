import torch
from src.slices.ExtractFeatures.SpeedPerturb_Transform import apply_speed_perturb
from src.shared_kernel.Config_Adapter import get_config


def test_slow_factor_lengthens_audio():
    wave = torch.randn(get_config().audio.sample_rate)
    slowed = apply_speed_perturb(wave, 0.9)
    assert slowed.numel() > wave.numel()  # 0.9x speed -> more samples
