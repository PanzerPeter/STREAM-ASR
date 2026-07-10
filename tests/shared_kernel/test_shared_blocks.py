import torch

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn
from src.shared_kernel.RoPE_Transform import rotary_tables, apply_rotary


def test_shared_blocks_forward_shapes():
    x = torch.randn(2, 5, 64)
    assert BiasNorm(64)(x).shape == (2, 5, 64)
    assert SwiGluFfn(64, expansion=4, dropout=0.0)(x).shape == (2, 5, 64)


def test_rope_apply_preserves_shape_and_norm():
    q = torch.randn(1, 4, 7, 16)
    cos, sin = rotary_tables(7, 16, q.device, q.dtype)
    out = apply_rotary(q, cos, sin)
    assert out.shape == q.shape
    # RoPE is a rotation -> per-position vector norm is preserved.
    torch.testing.assert_close(out.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-4)
