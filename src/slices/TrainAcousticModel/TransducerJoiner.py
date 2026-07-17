import torch
import torch.nn as nn

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

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # enc [B, T, De], pred [B, U', Dp] -> [B, T, U', V] via broadcast over the (T, U') grid.
        e = self.enc_proj(enc).unsqueeze(2)  # [B, T, 1, J]
        p = self.pred_proj(pred).unsqueeze(1)  # [B, 1, U', J]
        return self.out(torch.tanh(e + p))

    def step(self, enc_t: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
        # enc_t [B, De], pred_u [B, Dp] -> [B, V] for a single decode cell.
        return self.out(torch.tanh(self.enc_proj(enc_t) + self.pred_proj(pred_u)))
