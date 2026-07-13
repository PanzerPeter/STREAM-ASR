import pytest
import torch

from src.shared_kernel.Muon_Optimizer import Muon, newton_schulz_orthogonalize


def test_newton_schulz_flattens_singular_spectrum():
    # Muon's Newton-Schulz is APPROXIMATE spectral normalization: it collapses a matrix's
    # singular-value spectrum toward 1 (spectrally normalizing every 2D update), not exact
    # orthogonalization — the min singular value floors around 0.68 with the 5-step Jordan
    # quintic. Validate that real contract: the raw matrix's wide, ill-conditioned spectrum is
    # flattened into a tight band near 1 with far better conditioning.
    torch.manual_seed(0)
    g = torch.randn(64, 32)
    sv_g = torch.linalg.svdvals(g.float())
    u = newton_schulz_orthogonalize(g, steps=5)
    sv_u = torch.linalg.svdvals(u)
    assert sv_u.max() < 1.3 and sv_u.min() > 0.5  # flattened into a tight band near 1
    assert sv_u.max() / sv_u.min() < (sv_g.max() / sv_g.min()) / 2  # conditioning sharply improved
    gram = u.t() @ u
    assert (gram - torch.eye(32)).abs().max() < 0.5  # approximately orthonormal (Jordan band)


def test_muon_step_reduces_quadratic_loss():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt = Muon([w], lr=0.05, momentum=0.9, ns_steps=5)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


def test_muon_rejects_non_2d_params():
    # Lock the 2D-only contract: routing a bias/1D (or any non-matrix) param to Muon must raise,
    # not silently mis-orthogonalize — non-2D params belong on AdamW (Task 7's partition
    # enforces this).
    w = torch.nn.Parameter(torch.randn(8))  # 1D
    (w * 2).sum().backward()  # give it a real gradient so step() reaches the ndim check
    opt = Muon([w], lr=0.01)
    with pytest.raises(ValueError):
        opt.step()
