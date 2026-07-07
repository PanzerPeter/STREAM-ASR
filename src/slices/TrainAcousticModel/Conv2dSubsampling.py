# src/slices/TrainAcousticModel/Conv2dSubsampling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


class Conv2dSubsampling(nn.Module):
    """×2 time subsampling conv frontend: [B,T,80] -> [B, (T-1)//2+1, out_dim]. Causal in time —
    the time dimension is left-padded only, so an output frame reads no future input. Frequency
    padding stays symmetric (frequency is not streamed). conv1 strides time ×2 + freq ×2; conv2
    strides frequency ×2 only (time stride 1)."""

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
        self.time_pad = 2  # kernel-1 causal window for both convs (conv1 stride 2, conv2 stride 1)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=(0, 1))  # time pad 0
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=(1, 2), padding=(0, 1))
        self.relu = nn.ReLU()
        self.freq_mid = (n_mels + 2 * 1 - 3) // 2 + 1  # freq bins after conv1 (80 -> 40)
        freq_after = (self.freq_mid + 2 * 1 - 3) // 2 + 1  # after conv2 (40 -> 20)
        self.linear = nn.Linear(channels * freq_after, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.relu(self.conv1(F.pad(x, (0, 0, self.time_pad, 0))))  # causal time left-pad
        x = self.relu(self.conv2(F.pad(x, (0, 0, self.time_pad, 0))))
        b, c, t2, f2 = x.shape
        x = x.transpose(1, 2).reshape(b, t2, c * f2)  # [B, T2, C*F2]
        x = self.linear(x)
        out_lengths = (lengths - 1) // 2 + 1
        return x, out_lengths

    def streaming_forward(
        self, x: torch.Tensor, in_tail: torch.Tensor, mid_tail: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Two-level causal cache: in_tail feeds conv1's left window, mid_tail feeds conv2's.
        # Start as zeros to reproduce sequence-start left-pad, streaming is exact from frame 0.
        if x.shape[1] < 2 * self.time_pad:
            # Below this a chunk yields fewer post-conv1 frames than mid_tail needs, silently
            # under-filling the cache and crashing conv2 two chunks later; fail clearly instead.
            raise ValueError(
                f"streaming chunk must be >= {2 * self.time_pad} frames, got {x.shape[1]}"
            )
        xin = torch.cat([in_tail, x], dim=1)  # [B, time_pad+Tc, F]
        new_in_tail = x[:, -self.time_pad :].detach()
        h = self.relu(self.conv1(xin.unsqueeze(1)))  # [B, C, T1, f_mid]
        mid = torch.cat([mid_tail, h], dim=2)  # concat along time
        new_mid_tail = h[:, :, -self.time_pad :, :].detach()
        y = self.relu(self.conv2(mid))  # [B, C, T2, f2]
        b, c, t2, f2 = y.shape
        y = y.transpose(1, 2).reshape(b, t2, c * f2)
        y = self.linear(y)
        return y, new_in_tail, new_mid_tail
