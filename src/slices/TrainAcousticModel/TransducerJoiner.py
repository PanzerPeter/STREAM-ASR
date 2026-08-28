import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


class TransducerJoiner(nn.Module):
    """Standard additive RNN-T joiner: project encoder memory and predictor output into a shared
    joiner space, sum, tanh, then a readout to the vocab+blank width. Training materialises the
    full [B, T, U+1, V] lattice for the RNN-T loss; decoding evaluates one (t, u) cell at a time."""

    def __init__(self) -> None:
        super().__init__()
        model = get_config().model
        t = get_config().transducer
        self.enc_proj = nn.Linear(model.encoder_dims[-1], t.joiner_dim)
        self.pred_proj = nn.Linear(t.predictor_dim, t.joiner_dim)
        self.out = nn.Linear(t.joiner_dim, model.logits_width)
        # The readout is the model's largest GEMM by a wide margin (it is the only one that runs
        # over the whole [B, T, U+1] lattice), and its N is logits_width = vocab + blank = 501.
        # A bf16 tensor-core GEMM needs its N stride 16-byte aligned to reach the fast kernels;
        # 501 * 2 bytes is not, so cuBLAS picks an alignment-2 fallback that measured 29 TFLOPS
        # against 57 at N = 504. Rounding N up to a multiple of 8 is worth 14% of the whole
        # training step. The padded columns carry a -inf bias, so exp() of their logit is exactly
        # 0: every log-softmax, gather and softmax downstream is bit-for-bit the 501-wide result,
        # their gradient is exactly 0, and the parameters stay 501-wide so checkpoints are
        # unaffected. Kept out of `step()`, which is one cell wide and GEMM-bound by nothing.
        self._pad = -model.logits_width % 8

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(
            h,
            F.pad(self.out.weight, (0, 0, 0, self._pad)),
            F.pad(self.out.bias, (0, self._pad), value=float("-inf")),
        )

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # enc [B, T, De], pred [B, U', Dp] -> [B, T, U', V] via broadcast over the (T, U') grid.
        e = self.enc_proj(enc).unsqueeze(2)  # [B, T, 1, J]
        p = self.pred_proj(pred).unsqueeze(1)  # [B, 1, U', J]
        return self._readout(torch.tanh(e + p))

    def band(self, enc: torch.Tensor, pred_band: torch.Tensor) -> torch.Tensor:
        # Pruned objective: enc [B, T, De], pred_band [B, T, S, Dp] -> [B, T, S, V]. The predictor
        # side is already gathered per frame, so unlike `forward` it does NOT broadcast over the
        # symbol axis -- each frame sees its own S-wide slice of the prediction network.
        e = self.enc_proj(enc).unsqueeze(2)  # [B, T, 1, J]
        p = self.pred_proj(pred_band)  # [B, T, S, J]
        return self._readout(torch.tanh(e + p))

    def step(self, enc_t: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
        # enc_t [B, De], pred_u [B, Dp] -> [B, V] for a single decode cell.
        return self.out(torch.tanh(self.enc_proj(enc_t) + self.pred_proj(pred_u)))
