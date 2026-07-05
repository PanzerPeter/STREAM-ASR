# src/slices/TrainAcousticModel/Resample.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDownsample(nn.Module):
    """Learnable weighted pooling over a `factor`-frame window (softmax weights).
    Downsamples the multi-rate Zipformer stack input; ceil semantics on lengths."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.weights = nn.Parameter(torch.zeros(factor))  # softmax(0)=uniform at init

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        b, t, c = x.shape
        f = self.factor
        pad = (f - t % f) % f
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # pad time so it divides evenly
        x = x.view(b, (t + pad) // f, f, c)
        w = torch.softmax(self.weights, dim=0).view(1, 1, f, 1)
        y = (x * w).sum(dim=2)  # [B, T/f, C]
        out_lengths = (lengths + f - 1) // f  # ceil
        return y, out_lengths


class SimpleUpsample(nn.Module):
    """Nearest-neighbour upsample by repeat, trimmed back to the pre-downsample length."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor, out_len: int) -> torch.Tensor:
        y = x.repeat_interleave(self.factor, dim=1)
        return y[:, :out_len]
