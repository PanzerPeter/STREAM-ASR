import torch
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Config_Adapter import get_config

_AUDIO = get_config().audio


def test_log_mel_shape_for_one_second():
    wave = torch.randn(_AUDIO.sample_rate)  # 1.0 s
    mel = compute_log_mel(wave)
    assert mel.shape[1] == _AUDIO.n_mels
    expected_frames = _AUDIO.sample_rate // _AUDIO.hop_length  # ~100
    assert abs(mel.shape[0] - expected_frames) <= 2


def test_log_mel_is_finite():
    wave = torch.randn(_AUDIO.sample_rate // 2)
    mel = compute_log_mel(wave)
    assert torch.isfinite(mel).all()
