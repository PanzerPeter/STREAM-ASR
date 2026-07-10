# src/slices/TrainLanguageModel/CausalGqaAttention.py
# Causal grouped-query self-attention for STREAM-LM. QK-norm stabilizes the deep-narrow stack;
# value-residual injects layer-0 values into deeper layers. step() is the incremental-decode path
# used by shallow fusion; it is exact w.r.t. the parallel forward().
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.RoPE_Transform import apply_rotary, rotary_tables


@dataclass
class KvCache:
    k: torch.Tensor  # [B, kv_groups, S, head_dim] (RoPE-embedded)
    v: torch.Tensor  # [B, kv_groups, S, head_dim] (post value-residual)
    seen: int

    @staticmethod
    def empty(
        batch: int,
        kv_groups: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KvCache":
        z = torch.zeros(batch, kv_groups, 0, head_dim, device=device, dtype=dtype)
        return KvCache(k=z, v=z.clone(), seen=0)


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    # QK-norm: normalize each head vector to unit RMS before the dot product (no affine — the
    # subsequent 1/sqrt(head_dim) SDPA scale absorbs magnitude).
    return x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()


class CausalGqaAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, kv_groups: int, dropout: float) -> None:
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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_groups * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.lam = 0.0  # set by StreamLmModel per layer (0 on layer 0, config value deeper)

    def _project(
        self, x: torch.Tensor, pos_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(b, t, 2, self.kv_groups, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        cos, sin = rotary_tables(t, self.head_dim, x.device, x.dtype, pos_offset)
        q, k = _rms_norm(q), _rms_norm(k)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        return q, k, v

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        kx = k.repeat_interleave(self.rep, dim=1)  # broadcast KV heads to query heads
        vx = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(
            q,
            kx,
            vx,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        b = out.shape[0]
        out = out.transpose(1, 2).reshape(b, out.shape[2], self.heads * self.head_dim)
        return self.out(out)

    def forward(
        self, x: torch.Tensor, value_residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self._project(x, pos_offset=0)
        if value_residual is not None:
            v = v + self.lam * value_residual
        out = self._attend(q, k, v, is_causal=True)
        return out, v

    def step(
        self, x_t: torch.Tensor, cache: KvCache, value_residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, KvCache]:
        q, k, v = self._project(x_t, pos_offset=cache.seen)
        if value_residual is not None:
            v = v + self.lam * value_residual
        k = torch.cat([cache.k, k], dim=2)
        v = torch.cat([cache.v, v], dim=2)
        out = self._attend(q, k, v, is_causal=False)  # single query attends all cached keys
        new_cache = KvCache(k=k.detach(), v=v.detach(), seen=cache.seen + x_t.shape[1])
        return out, v[:, :, -x_t.shape[1] :], new_cache
