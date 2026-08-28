import torch

from src.shared_kernel.RnntLoss import rnnt_loss
from src.shared_kernel.RnntLossPruned import prune_ranges, rnnt_loss_pruned, rnnt_loss_simple


def _fixture(b=2, t=6, u=3, v=7, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    blank = v - 1
    am = torch.randn(b, t, v, dtype=dtype, requires_grad=True)
    lm = torch.randn(b, u + 1, v, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, blank, (b, u), dtype=torch.int32)
    # Ragged in both axes, so padding bugs cannot hide behind a square batch.
    frames = torch.tensor([max(1, t - i) for i in range(b)], dtype=torch.int32)
    labels = torch.tensor([max(0, u - i) for i in range(b)], dtype=torch.int32)
    return am, lm, targets, frames, labels, blank


def test_simple_loss_matches_the_explicit_additive_lattice():
    # The simple joiner IS logits[b,t,u,v] = am[b,t,v] + lm[b,u,v]. Building that lattice explicitly
    # and running the reference loss must give the same cost -- otherwise the pruning bounds are
    # derived from a model the training objective does not contain.
    am, lm, targets, frames, labels, blank = _fixture()
    cost, occupancy = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)

    logits = am.unsqueeze(2) + lm.unsqueeze(1)  # [B, T, U+1, V]
    reference = rnnt_loss(logits, targets, frames, labels, blank=blank, reduction="none")

    assert torch.allclose(cost, reference, atol=1e-9)
    assert occupancy.shape == (am.shape[0], am.shape[1], lm.shape[1])
    assert not occupancy.requires_grad  # bounds are integers; they carry no gradient


def test_simple_loss_gradient_matches_autograd_through_the_explicit_lattice():
    # The analytic gradient is the whole reason this is a custom Function -- the [B,T,U+1,V] tensor
    # is recomputed in backward rather than stored. Lock it against plain autograd.
    am, lm, targets, frames, labels, blank = _fixture()
    cost, _ = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)
    cost.sum().backward()
    got_am, got_lm = am.grad.clone(), lm.grad.clone()

    am2 = am.detach().clone().requires_grad_(True)
    lm2 = lm.detach().clone().requires_grad_(True)
    logits = am2.unsqueeze(2) + lm2.unsqueeze(1)
    rnnt_loss(logits, targets, frames, labels, blank=blank, reduction="none").sum().backward()

    assert torch.allclose(got_am, am2.grad, atol=1e-9)
    assert torch.allclose(got_lm, lm2.grad, atol=1e-9)


def test_simple_loss_occupancy_is_a_distribution_over_each_frame():
    # occupancy[b,t,:] is the posterior over symbol positions at frame t. It must sum to 1 across u
    # within each valid frame, or the expected-position bound generator is reading a mis-normalised
    # quantity. Note this is the BLANK-ARC posterior: exp(alpha + beta - Z) counts every node the
    # path visits, and a path visits several u at one t, so that one sums well above 1.
    am, lm, targets, frames, labels, blank = _fixture()
    _, occupancy = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)
    for b, n_frames in enumerate(frames.tolist()):
        sums = occupancy[b, :n_frames].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_prune_ranges_satisfy_every_reachability_constraint():
    # A band set that violates any of these makes the pruned lattice contain no complete path, and
    # the loss comes back -inf. Assert all four directly rather than inferring them from a NaN.
    am, lm, targets, frames, labels, blank = _fixture(b=3, t=12, u=5, v=9, seed=7)
    _, occupancy = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)
    s_range = 3
    s_begin = prune_ranges(occupancy, frames.long(), labels.long(), s_range=s_range)

    assert s_begin.shape == (3, 12)
    assert s_begin.dtype == torch.int64
    for b, (n_frames, n_labels) in enumerate(zip(frames.tolist(), labels.tolist())):
        row = s_begin[b, :n_frames]
        assert row[0].item() == 0, "band must contain the start state (0, 0)"
        assert row[-1].item() >= n_labels + 1 - s_range, "band must contain the accepting state"
        assert torch.all(row[1:] >= row[:-1]), "band starts must be monotone non-decreasing"
        assert torch.all(row[1:] - row[:-1] <= s_range - 1), "consecutive bands must overlap"
        assert torch.all(row >= 0) and torch.all(row <= n_labels + 1 - s_range)


def test_prune_ranges_degenerate_when_the_band_covers_everything():
    # U+1 <= s_range means there is nothing to prune: every band must start at 0.
    am, lm, targets, frames, labels, blank = _fixture(b=2, t=5, u=2, v=6, seed=3)
    _, occupancy = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)
    s_begin = prune_ranges(occupancy, frames.long(), labels.long(), s_range=8)
    assert torch.all(s_begin == 0)


def _feasible_s_begin(frames, labels, s_range, num_frames):
    # Monotone unit-step ramp reaching top = labels+1-s_range exactly at each row's last valid
    # frame: the cheapest s_begin that satisfies all four reachability constraints, built without
    # going through prune_ranges so these tests do not depend on it.
    top = (labels.long() + 1 - s_range).clamp(min=0)
    last = (frames.long() - 1).clamp(min=0)
    t = torch.arange(num_frames).view(1, -1)
    return (t - (last - top).view(-1, 1)).clamp(min=0).minimum(top.view(-1, 1))


def test_pruned_with_a_full_width_band_is_exactly_the_full_loss():
    # THE load-bearing test. s_range = U+1 with every band at 0 keeps every alignment, so the
    # pruned recursion must reproduce rnnt_loss bit-for-bit in fp64 -- forward AND gradient. A
    # failure here means the band scan's shift arithmetic is wrong, which no WER curve would
    # reveal until a 50 h run had already finished.
    torch.manual_seed(11)
    b, t, u, v, blank = 3, 9, 4, 8, 7
    logits = torch.randn(b, t, u + 1, v, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, blank, (b, u), dtype=torch.int32)
    frames = torch.tensor([9, 7, 8], dtype=torch.int32)
    labels = torch.tensor([4, 3, 2], dtype=torch.int32)
    s_begin = torch.zeros(b, t, dtype=torch.int64)

    got = rnnt_loss_pruned(logits, targets, s_begin, frames, labels, blank=blank, reduction="none")
    got.sum().backward()
    got_grad = logits.grad.clone()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = rnnt_loss(ref_logits, targets, frames, labels, blank=blank, reduction="none")
    ref.sum().backward()

    assert torch.allclose(got, ref, atol=1e-9)
    assert torch.allclose(got_grad, ref_logits.grad, atol=1e-9)


def test_pruned_loss_is_an_upper_bound_on_the_full_loss():
    # Pruning drops alignments, so it can only raise -log P. A pruned cost BELOW the full one means
    # the recursion is double-counting a path.
    torch.manual_seed(5)
    b, t, u, v, blank = 2, 10, 5, 9, 8
    full = torch.randn(b, t, u + 1, v, dtype=torch.float64)
    targets = torch.randint(0, blank, (b, u), dtype=torch.int32)
    frames = torch.tensor([10, 9], dtype=torch.int32)
    labels = torch.tensor([5, 4], dtype=torch.int32)

    s_range = 3
    am = torch.zeros(b, t, v, dtype=torch.float64)
    lm = torch.zeros(b, u + 1, v, dtype=torch.float64)
    _, occupancy = rnnt_loss_simple(am, lm, targets, frames, labels, blank=blank)
    s_begin = prune_ranges(occupancy, frames.long(), labels.long(), s_range=s_range)

    band = torch.stack(
        [
            full[bi, ti, s_begin[bi, ti] : s_begin[bi, ti] + s_range]
            for bi in range(b)
            for ti in range(t)
        ]
    ).view(b, t, s_range, v)

    pruned = rnnt_loss_pruned(band, targets, s_begin, frames, labels, blank=blank, reduction="none")
    reference = rnnt_loss(full, targets, frames, labels, blank=blank, reduction="none")
    assert torch.all(pruned >= reference - 1e-9)
    assert torch.all(torch.isfinite(pruned))


def test_pruned_handles_an_empty_transcript():
    # An empty transcript has exactly one alignment: all blanks. Its cost is -sum log p(blank).
    # torchaudio's CUDA kernel returns 0.0 here; this repo's loss does not, nor may this one.
    torch.manual_seed(2)
    b, t, v, blank = 1, 4, 6, 5
    logits = torch.randn(b, t, 1, v, dtype=torch.float64)
    targets = torch.zeros(b, 0, dtype=torch.int32)
    frames = torch.tensor([4], dtype=torch.int32)
    labels = torch.tensor([0], dtype=torch.int32)
    s_begin = torch.zeros(b, t, dtype=torch.int64)

    got = rnnt_loss_pruned(logits, targets, s_begin, frames, labels, blank=blank, reduction="none")
    expected = -torch.log_softmax(logits, dim=-1)[0, :, 0, blank].sum()
    assert torch.allclose(got[0], expected, atol=1e-9)


def test_pruned_ignores_padding_past_each_utterance_length():
    # Batches are ragged. Rewriting a padded region must not move the cost, or a batch's loss
    # depends on who it was bucketed with.
    torch.manual_seed(13)
    b, t, u, v, blank = 2, 8, 3, 7, 6
    s_range = 2
    logits = torch.randn(b, t, s_range, v, dtype=torch.float64)
    targets = torch.randint(0, blank, (b, u), dtype=torch.int32)
    frames = torch.tensor([8, 5], dtype=torch.int32)
    labels = torch.tensor([3, 2], dtype=torch.int32)
    s_begin = _feasible_s_begin(frames, labels, s_range, t)

    before = rnnt_loss_pruned(
        logits, targets, s_begin, frames, labels, blank=blank, reduction="none"
    )
    logits[1, 5:] = torch.randn(t - 5, s_range, v, dtype=torch.float64)
    after = rnnt_loss_pruned(
        logits, targets, s_begin, frames, labels, blank=blank, reduction="none"
    )
    assert torch.allclose(before, after, atol=1e-9)
