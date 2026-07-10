import torch

from src.slices.TrainLanguageModel.CausalGqaAttention import CausalGqaAttention, KvCache


def _module():
    torch.manual_seed(0)
    return CausalGqaAttention(d_model=32, heads=4, kv_groups=2, dropout=0.0).eval()


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


def test_incremental_step_matches_full_forward():
    m = _module()
    x = torch.randn(1, 5, 32)
    full, _ = m(x, value_residual=None)
    cache = KvCache.empty(batch=1, kv_groups=2, head_dim=8, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(5):
        o, _, cache = m.step(x[:, t : t + 1], cache, value_residual=None)
        outs.append(o)
    inc = torch.cat(outs, dim=1)
    torch.testing.assert_close(full, inc, atol=2e-5, rtol=2e-5)
