# src/slices/TrainAcousticModel/RotaryAttention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


def _rotary_tables(seq_len: int, head_dim: int, device, dtype):
    # Standard RoPE: pair up channels and rotate by position-dependent angles.
    rope_base = get_config().model.rope_base
    half = head_dim // 2
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)  # [T, half]
    emb = torch.cat([angles, angles], dim=-1)  # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class RotaryAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings. RoPE is applied to q/k
    before the fused SDPA kernel (flash / mem-efficient under bf16 autocast), so the
    [B, H, T, T] score matrix is never materialized. SDPA's default scale (1/sqrt(head_dim))
    and its attn-weight dropout match the previous manual matmul→mask→softmax→matmul path."""

    def __init__(self, dim: int, num_heads: int, dropout: float | None = None) -> None:
        super().__init__()
        if dropout is None:
            dropout = get_config().model.encoder_dropout
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor, attn_visible: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, T, D]

        cos, sin = _rotary_tables(t, self.head_dim, x.device, x.dtype)
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)

        # True = attend. Start from the padding mask (True at pad -> invert), then AND the
        # chunk-visibility mask [T, T] when streaming-style masking is requested.
        attn_mask = ~pad_mask[:, None, None, :]  # [B, 1, 1, T]
        if attn_visible is not None:
            attn_mask = attn_mask & attn_visible[None, None, :, :]  # [B, 1, T, T]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.out_dropout(self.out(out))
