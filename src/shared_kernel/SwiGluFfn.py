# src/shared_kernel/SwiGluFfn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


def _rounded_hidden(dim: int, expansion: int) -> int:
    hidden = int(dim * expansion * 2 / 3)  # 2/3 keeps SwiGLU param-count ~= a dense FFN
    return (hidden + 7) // 8 * 8  # round to a multiple of 8 for tensor-core alignment


class SwiGluFfn(nn.Module):
    def __init__(
        self, dim: int, expansion: int | None = None, dropout: float | None = None
    ) -> None:
        super().__init__()
        cfg = get_config().model
        expansion = cfg.ffn_expansion if expansion is None else expansion
        dropout = cfg.encoder_dropout if dropout is None else dropout
        hidden = _rounded_hidden(dim, expansion)
        self.w_gate = nn.Linear(dim, hidden)
        self.w_up = nn.Linear(dim, hidden)
        self.w_down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
