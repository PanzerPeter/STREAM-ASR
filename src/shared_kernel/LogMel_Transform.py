# src/shared_kernel/LogMel_Transform.py — pure feature transform (Shared Kernel eligible)
import torch
import torchaudio

from src.shared_kernel.Config_Adapter import get_config

_LOG_EPS = 1e-10  # floor before log to keep silence finite

_audio = get_config().audio
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=_audio.sample_rate,
    n_fft=_audio.n_fft,
    win_length=_audio.win_length,
    hop_length=_audio.hop_length,
    n_mels=_audio.n_mels,
    power=2.0,
)


def compute_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 1:
        raise ValueError(f"expected 1-D waveform, got shape {tuple(waveform.shape)}")

    mel = _MEL(waveform)  # [N_MELS, T]
    log_mel = torch.log(mel + _LOG_EPS)  # log-compression; floor avoids -inf on silence
    return log_mel.transpose(0, 1).contiguous()  # -> [T, N_MELS], time-major for sequence models
