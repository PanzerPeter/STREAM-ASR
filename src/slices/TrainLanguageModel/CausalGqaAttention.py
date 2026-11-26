# Causal grouped-query self-attention for STREAM-LM: QK-norm + RoPE, and a learnable value residual
# (layer 0's values mixed into every deeper layer, Zhou et al. 2024, "Value Residual Learning").
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.RoPE_Transform import apply_rotary, rope_inv_freq, rotary_tables

_EPS = 1e-6


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    # Weightless: no per-channel gain to grow, the same normaliser the acoustic stack uses.
    return F.rms_norm(x, (x.shape[-1],), eps=_EPS)


class CausalGqaAttention(nn.Module):
    inv_freq: torch.Tensor
    value_lambda: nn.Parameter | None

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_groups: int,
        dropout: float,
        value_residual: float | None,
    ) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        if heads % kv_groups != 0:
            raise ValueError(f"heads {heads} not divisible by kv_groups {kv_groups}")
        self.heads = heads
        self.kv_groups = kv_groups
        self.head_dim = d_model // heads
        self.rep = heads // kv_groups
        self.dropout_p = dropout
        # K and V are separate matrices so Muon orthogonalises each on its own scale; one fused
        # [k; v] matrix would share a single spectral normalisation between them.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_groups * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        # None on layer 0, which produces the residual values; a learnable scalar deeper.
        self.value_lambda = (
            nn.Parameter(torch.tensor(float(value_residual)))
            if value_residual is not None
            else None
        )
        inv_freq = rope_inv_freq(self.head_dim, get_config().model.rope_base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)
        cos, sin = rotary_tables(torch.arange(t, device=x.device), self.inv_freq, x.dtype)
        # QK-norm: unit-RMS head vectors bound every attention logit by sqrt(head_dim).
        q, k = rms_norm(q), rms_norm(k)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        return q, k, v

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Explicit KV broadcast rather than SDPA's enable_gqa: with a boolean document mask the
        # fused kernels that take enable_gqa are not eligible, and the copy is 2 -> 8 heads at
        # T=256.
        kx = k.repeat_interleave(self.rep, dim=1)
        vx = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(
            q,
            kx,
            vx,
            attn_mask=attn_mask,
            # SDPA rejects is_causal together with an explicit mask; the caller's mask is already
            # causal (it is the causal mask ANDed with the same-line constraint).
            is_causal=attn_mask is None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        b = out.shape[0]
        out = out.transpose(1, 2).reshape(b, out.shape[2], self.heads * self.head_dim)
        return self.out(out)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self._project(x)
        if self.value_lambda is not None and value_residual is not None:
            # A 0-dim fp32 scalar times a bf16 tensor stays bf16 under autocast.
            v = v + self.value_lambda * value_residual
        return self._attend(q, k, v, attn_mask), v
