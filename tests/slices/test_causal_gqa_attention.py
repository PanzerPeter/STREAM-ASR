import torch

from src.slices.TrainLanguageModel.CausalGqaAttention import CausalGqaAttention


def _module(value_residual: float | None = None) -> CausalGqaAttention:
    torch.manual_seed(0)
    return CausalGqaAttention(
        d_model=32, heads=4, kv_groups=2, dropout=0.0, value_residual=value_residual
    ).eval()


def test_forward_shapes_and_gqa():
    m = _module()
    x = torch.randn(2, 6, 32)
    out, v = m(x, value_residual=None)
    assert out.shape == (2, 6, 32)
    assert v.shape == (2, 2, 6, 8)  # kv_groups=2, head_dim=8


def test_causality_future_tokens_do_not_change_past():
    m = _module()
    x = torch.randn(1, 6, 32)
    out_a, _ = m(x, value_residual=None)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 32)  # perturb the future
    out_b, _ = m(x2, value_residual=None)
    torch.testing.assert_close(out_a[:, :4], out_b[:, :4], atol=1e-5, rtol=1e-5)


def test_value_residual_mixes_layer_zero_values_by_the_learned_weight():
    m = _module(value_residual=0.5)
    x = torch.randn(1, 6, 32)
    v0 = torch.randn(1, 2, 6, 8)
    with torch.no_grad():
        _, v_plain = _module(value_residual=None)(x, value_residual=None)
        _, v_mixed = m(x, value_residual=v0)
    torch.testing.assert_close(v_mixed, v_plain + 0.5 * v0)
