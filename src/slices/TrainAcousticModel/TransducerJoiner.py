import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


class TransducerJoiner(nn.Module):
    """Additive RNN-T joiner: project encoder memory and predictor output into a shared joiner
    space, sum, tanh, then a readout to vocab+blank. Training evaluates it on the full
    [B, T, U+1] lattice (band warmup) or on the band only; decoding one (t, u) cell at a time."""

    def __init__(self, encoder_dim: int) -> None:
        super().__init__()
        model = get_config().model
        t = get_config().transducer
        self.enc_proj = nn.Linear(encoder_dim, t.joiner_dim)
        self.pred_proj = nn.Linear(t.predictor_dim, t.joiner_dim)
        self.out = nn.Linear(t.joiner_dim, model.logits_width)
        # The readout is the step's largest GEMM and its N = 501 is not 16-byte aligned in bf16,
        # so cuBLAS falls back to an alignment-2 kernel (29 vs 57 TFLOPS at N = 504). Padding N to a
        # multiple of 8 with a -inf bias leaves every softmax bit-identical and the pad gradient 0;
        # parameters stay 501-wide. Not applied in `step`, which is one cell wide.
        self._pad = -model.logits_width % 8

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(
            h,
            F.pad(self.out.weight, (0, 0, 0, self._pad)),
            F.pad(self.out.bias, (0, self._pad), value=float("-inf")),
        )

    def forward(
        self, enc: torch.Tensor, pred: torch.Tensor, band_cols: torch.Tensor | None = None
    ) -> torch.Tensor:
        """enc [B, T, De], pred [B, U+1, Dp] -> [B, T, U+1, V'] or, with band_cols [B, T, W'] (the
        label state of each band cell), [B, T, W', V']. V' is the aligned readout width."""
        e = self.enc_proj(enc).unsqueeze(2)  # [B, T, 1, J]
        p = self.pred_proj(pred)  # [B, U+1, J]
        if band_cols is None:
            return self._readout(torch.tanh(e + p.unsqueeze(1)))
        b, t, w = band_cols.shape
        rows = p.gather(1, band_cols.reshape(b, t * w, 1).expand(-1, -1, p.shape[-1]))
        return self._readout(torch.tanh(e + rows.view(b, t, w, -1)))

    def step(self, enc_t: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
        # enc_t [..., De], pred_u [..., Dp] -> [..., V] for decode cells.
        return self.out(torch.tanh(self.enc_proj(enc_t) + self.pred_proj(pred_u)))
