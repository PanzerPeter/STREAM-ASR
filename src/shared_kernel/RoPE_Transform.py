# Pure RoPE tables + application (shared transform). The inverse frequencies are built once by the
# caller and held as a buffer, so a module's forward never reaches get_config() (that breaks a
# torch.compile graph).
import torch


def rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    half = head_dim // 2
    return 1.0 / (base ** (torch.arange(0, half).float() / half))


def rotary_tables(
    positions: torch.Tensor, inv_freq: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
