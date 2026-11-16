import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config

# Mel frames per 25 Hz output frame (two stride-2 stages).
FRONTEND_STRIDE = 4


class DwSepSubsampling(nn.Module):
    """x4 time / x4 freq depthwise-separable subsampling: [B, T, 80] -> [B, T/4, d] at 25 Hz,
    then a weightless RMSNorm so the encoder's first trunk operator emits RMS 1 per frame.

    Each stride-2 stage pads ONE frame on the left, not two. With two, output i of a k=3 stride-2
    conv reads inputs 2i-2..2i and stops at the first frame of its own pair, so stacked twice a
    25 Hz frame read mel 4j-6..4j and deferred 30 ms of its own 40 ms to the next frame. With one
    it reads 4j-3..4j+3: its whole group of four. The look-ahead never leaves the group and a
    streaming chunk is a whole number of groups, so this adds no latency and keeps streaming ==
    batched. Streaming state is one frame per stage (`in_tail`, `mid_tail`).
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        cfg = get_config()
        n_mels = cfg.audio.n_mels
        channels = cfg.model.frontend_channels
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=(0, 1))
        self.dw2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=(0, 1), groups=channels
        )
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.freq_mid = (n_mels - 1) // 2 + 1  # 80 -> 40 (k=3, pad 1, stride 2)
        freq_out = (self.freq_mid - 1) // 2 + 1  # 40 -> 20
        self.linear = nn.Linear(channels * freq_out, out_dim, bias=False)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.pw2(self.dw2(h)))  # [B, C, T1, F2]
        b, c, t1, f2 = y.shape
        y = self.linear(y.transpose(1, 2).reshape(b, t1, c * f2))
        return F.rms_norm(y, (y.shape[-1],), eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B, T, F], T a multiple of FRONTEND_STRIDE (the encoder pads it).
        h = F.relu(self.conv1(F.pad(x.unsqueeze(1), (0, 0, 1, 0))))
        return self._head(F.pad(h, (0, 0, 1, 0)))

    def streaming_forward(
        self, x: torch.Tensor, in_tail: torch.Tensor, mid_tail: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Zero-initialised tails reproduce forward's left pad: streaming is exact from frame 0.
        h = F.relu(self.conv1(torch.cat([in_tail, x], dim=1).unsqueeze(1)))
        y = self._head(torch.cat([mid_tail, h], dim=2))
        return y, x[:, -1:].detach(), h[:, :, -1:].detach()
