# Locks the banded RNN-T (MODEL_ARCHITECTURE.md §6.5) to the full loss and the band builder to the
# four properties the scan needs: start (0, 0), end (T-1, U), monotone, per-frame rise <= W'-1.
import torch

from src.shared_kernel.RnntLoss import _NEG, alignment_loss, occupancy_band, rnnt_loss


def _case(seed: int, batch=3, frames=12, labels=6, vocab=9):
    torch.manual_seed(seed)
    logits = torch.randn(batch, frames, labels + 1, vocab)
    targets = torch.randint(0, vocab - 1, (batch, labels), dtype=torch.int32)
    tlens = torch.tensor([labels, labels - 2, labels - 3], dtype=torch.int32)[:batch]
    flens = torch.tensor([frames, frames - 1, frames - 4], dtype=torch.int32)[:batch]
    return logits, targets, flens, tlens


def _band_logits(logits: torch.Tensor, lo: torch.Tensor, width: int) -> torch.Tensor:
    cols = lo.unsqueeze(2) + torch.arange(width)
    return logits.gather(2, cols.unsqueeze(3).expand(-1, -1, -1, logits.shape[3]))


def test_full_width_band_is_bit_identical_to_the_full_loss():
    # Invariant 7: at W' = U+1 the band IS the full lattice, the same code path, bit for bit.
    logits, targets, flens, tlens = _case(0)
    blank = logits.shape[3] - 1
    lo, width, _, infeasible = occupancy_band(
        torch.randn(3, logits.shape[1]).log_softmax(-1), flens, tlens, band_width=64
    )
    assert width == targets.shape[1] + 1 and lo.eq(0).all() and not infeasible.any()
    full_in = logits.clone().requires_grad_(True)
    band_in = logits.clone().requires_grad_(True)
    full = rnnt_loss(full_in, targets, flens, tlens, blank, reduction="none")
    band = rnnt_loss(band_in, targets, flens, tlens, blank, reduction="none", band_lo=lo)
    full.sum().backward()
    band.sum().backward()
    assert torch.equal(full, band) and torch.equal(full_in.grad, band_in.grad)


def test_band_equals_the_full_grid_with_out_of_band_arcs_removed():
    logits, targets, flens, tlens = _case(1, frames=14, labels=8)
    blank = logits.shape[3] - 1
    lo, width, _, _ = occupancy_band(torch.zeros(3, 14), flens, tlens, band_width=4)
    band = rnnt_loss(
        _band_logits(logits, lo, width), targets, flens, tlens, blank, "none", band_lo=lo
    )
    lp = logits.log_softmax(-1)
    u = torch.arange(lp.shape[2])
    inside = (u >= lo.unsqueeze(2)) & (u < lo.unsqueeze(2) + width)
    blank_lp = lp[..., blank].masked_fill(~inside, _NEG)
    label_lp = (
        lp[:, :, :-1]
        .gather(3, targets.long().view(3, 1, -1, 1).expand(-1, lp.shape[1], -1, 1))
        .squeeze(3)
    )
    label_lp = label_lp.masked_fill(~inside[:, :, :-1], _NEG)
    ref, _, _ = alignment_loss(blank_lp, label_lp, flens.long(), tlens.long())
    assert torch.allclose(band, ref, atol=1e-5)
    full = rnnt_loss(logits, targets, flens, tlens, blank, "none")
    assert (band >= full - 1e-5).all()  # a band is a subset of the paths


def test_gradcheck_float64_on_a_narrow_band():
    torch.manual_seed(2)
    batch, frames, labels, vocab = 2, 7, 5, 6
    targets = torch.randint(0, vocab - 1, (batch, labels), dtype=torch.int32)
    flens = torch.tensor([frames, frames - 1], dtype=torch.int32)
    tlens = torch.tensor([labels, labels - 1], dtype=torch.int32)
    lo, width, _, _ = occupancy_band(torch.zeros(batch, frames), flens, tlens, band_width=2)
    logits = torch.randn(batch, frames, width, vocab, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: rnnt_loss(x, targets, flens, tlens, vocab - 1, "none", band_lo=lo),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_late_burst_keeps_the_terminal_in_the_band():
    # §6.5's measured failure of a forward-only repair: U = 90, T = 150, the head's non-blank mass
    # in the last 3 frames. Without the reverse pass lo(T-1) = 48 against the 74 required, and the
    # cost is a finite 2e30 with an inf gradient.
    t, u, w = 150, 90, 16
    p_nb = torch.zeros(1, t)
    p_nb[0, -3:] = 0.97
    blank_lp = (1 - p_nb).clamp(min=1e-6).log()
    lo, width, _, infeasible = occupancy_band(
        blank_lp, torch.tensor([t]), torch.tensor([u]), band_width=w
    )
    assert not infeasible.any() and int(lo[0, -1]) == u + 1 - width
    torch.manual_seed(3)
    logits = torch.randn(1, t, width, 20, requires_grad=True)
    targets = torch.randint(0, 19, (1, u), dtype=torch.int32)
    cost = rnnt_loss(logits, targets, torch.tensor([t]), torch.tensor([u]), 19, band_lo=lo)
    cost.backward()
    assert torch.isfinite(cost) and float(cost.detach()) < 1e4
    assert torch.isfinite(logits.grad).all()


def test_band_properties_hold_on_adversarial_occupancies():
    torch.manual_seed(4)
    for _ in range(400):
        batch = 6
        frames = torch.randint(20, 120, (batch,))
        labels = (torch.rand(batch) * 0.95 * frames).long().clamp(min=1)
        t_max = int(frames.max())
        # Spiky, bursty, all-blank and all-emit heads, padded past each row's length.
        p_nb = torch.rand(batch, t_max) ** torch.randint(1, 8, (batch, 1)).float()
        p_nb[0] = 0.0
        p_nb[1] = 1.0
        p_nb[2, : t_max // 2] = 0.0
        blank_lp = (1 - p_nb).clamp(min=1e-6).log()
        w = int(torch.randint(1, 33, ()))
        lo, width, _, infeasible = occupancy_band(blank_lp, frames, labels, band_width=w)
        assert not infeasible.any()
        for b in range(batch):
            row = lo[b, : int(frames[b])]
            steps = row[1:] - row[:-1]
            assert int(row[0]) == 0
            assert int(row[-1]) == max(int(labels[b]) + 1 - width, 0)
            assert (steps >= 0).all() and (steps <= width - 1).all()
