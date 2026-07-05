# src/slices/TrainAcousticModel/Conv2dSubsampling.py
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config


class Conv2dSubsampling(nn.Module):
    """×2 time subsampling conv frontend: [B,T,80] -> [B, (T-1)//2+1, out_dim].
    Time stride 2 lives only in the first conv; the second conv strides frequency only."""

    def __init__(
        self,
        n_mels: int | None = None,
        channels: int | None = None,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        cfg = get_config()
        n_mels = cfg.audio.n_mels if n_mels is None else n_mels
        channels = cfg.model.frontend_channels if channels is None else channels
        out_dim = cfg.model.encoder_dims[0] if out_dim is None else out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),  # time ×2, freq ×2
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(1, 2), padding=1),  # freq ×2 only
            nn.ReLU(),
        )
        freq_after = ((n_mels + 1) // 2 + 1) // 2  # 80 -> 40 -> 20
        self.linear = nn.Linear(channels * freq_after, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.conv(x.unsqueeze(1))  # [B, C, T2, F2]
        b, c, t2, f2 = x.shape
        x = x.transpose(1, 2).reshape(b, t2, c * f2)  # [B, T2, C*F2]
        x = self.linear(x)  # [B, T2, out_dim]
        out_lengths = (lengths - 1) // 2 + 1
        return x, out_lengths
