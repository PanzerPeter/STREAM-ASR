# src/slices/TrainAcousticModel/ConvModule.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conformer/Zipformer depthwise conv module: pointwise GLU expand -> depthwise conv
    -> norm -> SiLU -> pointwise project. Padding positions are zeroed before the depthwise
    conv so they cannot leak into valid timesteps."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd for 'same' padding")
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=dim
        )  # batch-independent -> streaming-safe
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.glu(self.pointwise1(x), dim=1)  # [B, C, T]
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)  # stop padding bleeding through the conv
        x = self.depthwise(x)
        x = self.activation(self.norm(x))
        x = self.pointwise2(x)
        return x.transpose(1, 2)  # [B, T, C]
