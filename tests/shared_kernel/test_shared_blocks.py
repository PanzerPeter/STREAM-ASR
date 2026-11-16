import torch

from src.shared_kernel.RoPE_Transform import apply_rotary, rope_inv_freq, rotary_tables


def test_rope_apply_preserves_shape_and_norm():
    q = torch.randn(1, 4, 7, 16)
    cos, sin = rotary_tables(torch.arange(7), rope_inv_freq(16, 10000.0), q.dtype)
    out = apply_rotary(q, cos, sin)
    assert out.shape == q.shape
    # RoPE is a rotation -> per-position vector norm is preserved.
    torch.testing.assert_close(out.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_rope_scores_depend_only_on_relative_position():
    # The streaming ring cache re-assigns positions window-locally every chunk; that is exact only
    # because q.k under RoPE depends on i - j alone.
    inv = rope_inv_freq(16, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(i: int, j: int) -> torch.Tensor:
        cq, sq = rotary_tables(torch.tensor([i]), inv, q.dtype)
        ck, sk = rotary_tables(torch.tensor([j]), inv, k.dtype)
        return (apply_rotary(q, cq, sq) * apply_rotary(k, ck, sk)).sum()

    torch.testing.assert_close(score(5, 2), score(105, 102), atol=1e-4, rtol=1e-4)
