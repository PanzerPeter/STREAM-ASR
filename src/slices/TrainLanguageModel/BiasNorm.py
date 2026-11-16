import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config


class BiasNorm(nn.Module):
    """Zipformer BiasNorm as STREAM-LM uses it: RMS normalisation after removing a learned
    per-channel bias, rescaled by exp(log_scale) inside a fixed window. The window and the
    amplification floor are part of the shipped lm_best.pt's function (see config/lm.yaml), so
    this stays exactly what the LM was trained and evaluated with."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        lm = get_config().lm
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.eps = lm.norm_eps
        self.log_scale_min = lm.norm_log_scale_min
        self.log_scale_max = lm.norm_log_scale_max
        # Stored squared and inverted so forward is one multiply rather than a divide and a pow.
        self.inv_amp_sq = 1.0 / lm.norm_max_amplification**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dividing by rms(x - bias) but scaling x leaves the output RMS free through the bias;
        # flooring the normaliser at rms(x)/max_amplification bounds it.
        sq = x.pow(2).mean(dim=-1, keepdim=True)
        centred = (x - self.bias).pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.maximum(centred, sq * self.inv_amp_sq).add(self.eps).rsqrt()
        gain = self.log_scale.clamp(self.log_scale_min, self.log_scale_max).exp()
        return x * (inv_rms * gain)
