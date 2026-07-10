# src/slices/TrainAcousticModel/ConvModule.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.BiasNorm import BiasNorm


class ConvModule(nn.Module):
    """Conformer/Zipformer depthwise conv module: pointwise GLU expand -> causal depthwise conv
    -> per-frame BiasNorm -> SiLU -> pointwise project. The conv reads no future frames (left pad
    only) and the norm is per-frame, so a streaming chunk is bit-for-bit the full-sequence result.
    (GroupNorm-over-time, used previously, could not: its statistics span the whole sequence.)"""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, padding=0, groups=dim)
        self.norm = BiasNorm(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.glu(self.pointwise1(x), dim=1)  # [B, C, T]
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)  # keep padding out of the conv window
        x = F.pad(x, (self.kernel - 1, 0))  # causal: left context only, no future frames
        x = self.depthwise(x)  # [B, C, T]
        x = self.norm(x.transpose(1, 2))  # BiasNorm normalizes per frame over channels -> [B, T, C]
        x = self.activation(x).transpose(1, 2)  # [B, C, T]
        x = self.pointwise2(x)
        return x.transpose(1, 2)  # [B, T, C]

    def streaming_forward(
        self, x: torch.Tensor, cache_left: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cache_left holds the previous chunk's trailing kernel-1 glu-output frames, replacing the
        # zero left-pad the padded forward uses at sequence start. Per-frame norm makes this exact.
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.glu(self.pointwise1(x), dim=1)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C] to concat cache along time
        padded = torch.cat([cache_left, x], dim=1)
        new_left = padded[:, -(self.kernel - 1) :].detach()
        y = self.depthwise(padded.transpose(1, 2))  # valid conv over cache+chunk -> [B, C, T]
        y = self.norm(y.transpose(1, 2))  # [B, T, C]
        y = self.activation(y).transpose(1, 2)  # [B, C, T]
        y = self.pointwise2(y)
        return y.transpose(1, 2), new_left
