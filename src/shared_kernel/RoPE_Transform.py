# src/shared_kernel/RoPE_Transform.py — pure RoPE tables + application (shared transform)
import torch

from src.shared_kernel.Config_Adapter import get_config


def rotary_tables(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    pos_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Standard RoPE: pair channels, rotate by position-dependent angles. pos_offset starts
    # positions at an arbitrary index (streaming chunks / incremental decode).
    rope_base = get_config().model.rope_base
    half = head_dim // 2
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(pos_offset, pos_offset + seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
