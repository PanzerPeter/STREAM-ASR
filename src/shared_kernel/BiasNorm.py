# src/shared_kernel/BiasNorm.py
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config


class BiasNorm(nn.Module):
    """Zipformer BiasNorm: RMS normalization computed after removing a learned per-channel
    bias, then rescaled by exp(log_scale). Unlike LayerNorm it keeps a length degree of
    freedom, which Zipformer relies on."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.eps = get_config().audio.cmvn_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x - self.bias).pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.log_scale.exp()
