import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config


class BiasNorm(nn.Module):
    """Zipformer BiasNorm: RMS normalization computed after removing a learned per-channel
    bias, then rescaled by exp(log_scale). Unlike LayerNorm it keeps a length degree of
    freedom, which Zipformer relies on -- inside a bounded window, which it also relies on."""

    def __init__(self, num_channels: int, log_scale_max: float | None = None) -> None:
        super().__init__()
        cfg = get_config()
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.eps = cfg.audio.cmvn_eps
        # Read once, here. `forward` is one of the four modules handed to inductor, and a
        # `get_config()` call inside it breaks the graph (see _train_utils._HOT_MODULES).
        self.log_scale_min = cfg.model.biasnorm_log_scale_min
        # Overridable because the TRUNK normaliser lives on a different scale from the branch ones.
        # The branch ceiling of exp(1.0) = 2.72 is sized so no processed branch amplifies by more
        # than e; the trunk carries the residual stream itself, and the model that shipped 3.43 %
        # runs its five inter-stack trunks at RMS 2.68/2.24/1.51/2.69/2.25 -- the whole healthy
        # population sits just UNDER the branch ceiling, so reusing it would start three of five
        # trunk norms pinned on their bound. See `model.trunk_norm_log_scale_max`.
        self.log_scale_max = (
            cfg.model.biasnorm_log_scale_max if log_scale_max is None else log_scale_max
        )
        # Stored squared and inverted so `forward` is one multiply rather than a divide and a pow.
        self.inv_amp_sq = 1.0 / cfg.model.biasnorm_max_amplification**2

    @torch.no_grad()
    def project(self) -> None:
        """Pull the raw gain back inside its window. Trainers call this after every optimizer step.

        `forward` clamps the value it uses, so the bound is honoured no matter who calls this. What
        this adds is that the STORED parameter never leaves the window, which is what keeps it
        alive: `clamp`'s gradient is exactly 0 outside the bounds, so a gain the optimizer pushes
        past one would receive no gradient ever again and could never come back. Resting exactly ON
        a bound, gradient still flows. Same projected-gradient-descent argument as
        `ZipformerStack.project`, for the same reason.
        """
        self.log_scale.clamp_(self.log_scale_min, self.log_scale_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x / rms * exp(log_scale)` is two full-size elementwise passes over x and two full-size
        # temporaries, for one multiply's worth of work. There are six of these per Zipformer block
        # (96 per encoder forward), all bandwidth-bound, so the pass count *is* the cost. rsqrt and
        # the scalar fold move the reciprocal and the scale into the [.., 1] statistic, leaving a
        # single pass over x. Same value up to one rounding, on a normaliser.
        #
        # The normaliser is floored at `rms(x) / max_amplification`, which bounds the module's
        # output RMS at `max_amplification * exp(log_scale)`. WITHOUT IT THE GAIN BOUND BOUNDS
        # NOTHING: this divides by the RMS of `x - bias` but scales `x`, so the output RMS is
        # `exp(log_scale) * rms(x)/rms(x - bias)` and the second factor is a free parameter's
        # distance from the data. MEASURED on the 2026-08-21 run (dev utterance, 147 frames):
        #
        #   step                       43.2k    48.6k    54.0k    59.4k    64.8k
        #   per-frame amp, median       0.87     1.08    27.16     1.11     1.22
        #   per-frame amp, max          1.4      1.7     31.0     39.1     68.7
        #   output RMS                  1.68     3.18    66.96    40.24    63.84
        #
        # -- all of it with `log_scale` pinned at exactly its 1.0 ceiling, i.e. inside its bound
        # and emitting RMS 187 on the worst frame where the bound implies 2.72. `bias` barely
        # moved (|bias| 124.2 -> 121.4); what moved is the residual stream's DC component growing
        # into it (cos(DC, bias) 0.68 -> 0.92, |DC|/|bias| 0.49 -> 1.09), collapsing the
        # denominator. `eps` is 1e-5 and floors nothing at these scales.
        #
        # Sized to be inert in the trained regime rather than to trade against it: over 158,081
        # frame-module pairs on the healthy step43200 checkpoint, 99.99 % amplify by under 2x and
        # the single worst module reaches 6.7x, so the 4x cap changes that checkpoint's encoder
        # output by 3.4e-4 relative (2x would cost 1.3e-3, 8x exactly 0). Past the cap the
        # denominator no longer depends on `bias`, so the escape direction stops receiving
        # gradient, while the output stays proportional to `exp(log_scale)` so the gain keeps
        # training -- the same projected-gradient argument as the bound on log_scale itself.
        sq = x.pow(2).mean(dim=-1, keepdim=True)
        centred = (x - self.bias).pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.maximum(centred, sq * self.inv_amp_sq).add(self.eps).rsqrt()
        # The clamp is on the value USED, not only on the value stored, so a checkpoint written
        # before the bound existed -- or under a wider one -- still computes inside it.
        gain = self.log_scale.clamp(self.log_scale_min, self.log_scale_max).exp()
        return x * (inv_rms * gain)
