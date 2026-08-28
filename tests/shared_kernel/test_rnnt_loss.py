# Locks the hand-rolled RNN-T forward-backward against
# torchaudio's reference kernel (the implementation it replaces) and against first principles.
import pytest
import torch
import torchaudio.functional as AF

from src.shared_kernel.RnntLoss import rnnt_loss


def _case(batch, frames, labels, vocab, seed):
    torch.manual_seed(seed)
    logits = torch.randn(batch, frames, labels + 1, vocab)
    targets = torch.randint(0, vocab - 1, (batch, labels), dtype=torch.int32)
    target_lengths = torch.randint(1, labels + 1, (batch,), dtype=torch.int32)
    target_lengths[0] = labels  # torchaudio requires max(target_lengths) == the padded U
    logit_lengths = torch.maximum(
        torch.randint(1, frames + 1, (batch,), dtype=torch.int32), target_lengths + 1
    ).clamp(max=frames)
    logit_lengths[0] = frames
    return logits, targets, logit_lengths, target_lengths


@pytest.mark.parametrize(
    "batch,frames,labels,vocab",
    [(3, 9, 4, 7), (2, 6, 5, 11), (4, 17, 9, 23), (2, 30, 12, 40), (5, 40, 20, 60)],
)
def test_matches_torchaudio_loss_and_gradient(batch, frames, labels, vocab):
    # Same objective, same gradient: a checkpoint is interchangeable between the two kernels.
    logits, targets, logit_lengths, target_lengths = _case(batch, frames, labels, vocab, seed=0)
    weights = torch.arange(1, batch + 1, dtype=torch.float32)  # unequal upstream grads
    ref_in = logits.clone().requires_grad_(True)
    ours_in = logits.clone().requires_grad_(True)

    ref = AF.rnnt_loss(
        ref_in, targets, logit_lengths, target_lengths, blank=vocab - 1, reduction="none"
    )
    (ref * weights).sum().backward()
    ours = rnnt_loss(
        ours_in, targets, logit_lengths, target_lengths, blank=vocab - 1, reduction="none"
    )
    (ours * weights).sum().backward()

    # Tolerances are fp32 round-off, not slack: an algorithmic difference (a mis-seeded accepting
    # state, an unmasked padded arc) showed up at 1e-1 while developing this.
    assert torch.allclose(ref, ours, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ref_in.grad, ours_in.grad, atol=1e-4, rtol=1e-4)


def test_gradcheck_float64():
    # Independent of torchaudio: finite differences against the analytic backward.
    torch.manual_seed(1)
    batch, frames, labels, vocab = 2, 5, 3, 6
    logits = torch.randn(batch, frames, labels + 1, vocab, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, vocab - 1, (batch, labels), dtype=torch.int32)
    logit_lengths = torch.tensor([frames, frames - 1], dtype=torch.int32)
    target_lengths = torch.tensor([labels, labels - 1], dtype=torch.int32)
    assert torch.autograd.gradcheck(
        lambda x: rnnt_loss(
            x, targets, logit_lengths, target_lengths, blank=vocab - 1, reduction="none"
        ),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_empty_transcript_is_the_all_blank_path():
    # With no labels the only alignment is "blank on every frame", so the cost is exactly
    # -sum_t log p(blank | t). torchaudio's CUDA kernel returns 0.0 for this case; this
    # implementation does not, which is why the contract is pinned here rather than against it.
    torch.manual_seed(2)
    frames, vocab = 4, 5
    logits = torch.randn(1, frames, 1, vocab)
    cost = rnnt_loss(
        logits,
        torch.zeros(1, 0, dtype=torch.int32),
        torch.tensor([frames], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        blank=vocab - 1,
        reduction="none",
    )
    expected = -torch.log_softmax(logits[0, :, 0], dim=-1)[:, vocab - 1].sum()
    assert torch.allclose(cost[0], expected, atol=1e-5)


def test_padding_does_not_change_the_cost():
    # An utterance's cost must depend only on its own logit_length/target_length rectangle, so
    # padding the batch out with extra frames and labels leaves it untouched.
    torch.manual_seed(3)
    frames, labels, vocab = 6, 3, 9
    logits = torch.randn(1, frames, labels + 1, vocab)
    lengths = torch.tensor([4], dtype=torch.int32)
    targets = torch.randint(0, vocab - 1, (1, labels), dtype=torch.int32)
    kwargs = dict(blank=vocab - 1, reduction="none")
    full = rnnt_loss(logits, targets, lengths, torch.tensor([2], dtype=torch.int32), **kwargs)
    trimmed = rnnt_loss(
        logits[:, :4, :3].contiguous(),
        targets[:, :2].contiguous(),
        lengths,
        torch.tensor([2], dtype=torch.int32),
        **kwargs,
    )
    assert torch.allclose(full, trimmed, atol=1e-5)


def test_reduction_modes_agree():
    logits, targets, logit_lengths, target_lengths = _case(3, 8, 4, 7, seed=4)
    args = (targets, logit_lengths, target_lengths)
    per_utt = rnnt_loss(logits, *args, blank=6, reduction="none")
    assert torch.allclose(rnnt_loss(logits, *args, blank=6, reduction="sum"), per_utt.sum())
    assert torch.allclose(rnnt_loss(logits, *args, blank=6, reduction="mean"), per_utt.mean())


def test_rejects_bad_shape_and_reduction():
    logits, targets, logit_lengths, target_lengths = _case(2, 5, 3, 7, seed=5)
    with pytest.raises(ValueError):
        rnnt_loss(logits[0], targets, logit_lengths, target_lengths, blank=6)
    with pytest.raises(ValueError):
        rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=6, reduction="median")


def test_bfloat16_logits_give_a_bfloat16_gradient():
    # The joiner emits bf16 under autocast; the loss promotes internally and must hand back a
    # gradient in the joiner's own dtype, close to what the fp32 path produces.
    logits, targets, logit_lengths, target_lengths = _case(2, 10, 4, 9, seed=6)
    ref_in = logits.clone().requires_grad_(True)
    half_in = logits.bfloat16().clone().requires_grad_(True)
    rnnt_loss(ref_in, targets, logit_lengths, target_lengths, blank=8).backward()
    rnnt_loss(half_in, targets, logit_lengths, target_lengths, blank=8).backward()
    assert half_in.grad.dtype == torch.bfloat16
    assert torch.allclose(half_in.grad.float(), ref_in.grad, atol=2e-2)


def test_frame_slabbing_does_not_change_the_result(monkeypatch):
    # The two lattice-sized passes run a frame-slab at a time to bound their fp32 working set.
    # Production shapes take one slab, so force many and check the seams: a slab boundary must not
    # shift the log-softmax (it is per-cell over V) or drop a scatter into the label gradient.
    from src.shared_kernel import RnntLoss

    logits, targets, logit_lengths, target_lengths = _case(3, 12, 5, 9, seed=7)
    args = (targets, logit_lengths, target_lengths)
    whole_in = logits.clone().requires_grad_(True)
    slabbed_in = logits.clone().requires_grad_(True)

    whole = rnnt_loss(whole_in, *args, blank=8, reduction="none")
    whole.sum().backward()
    monkeypatch.setattr(RnntLoss, "_WORK_BYTES", 1)  # -> one frame per slab
    assert RnntLoss._frame_slab(3, 12, 6, 9) == 1
    slabbed = rnnt_loss(slabbed_in, *args, blank=8, reduction="none")
    slabbed.sum().backward()

    assert torch.equal(whole, slabbed)
    assert torch.equal(whole_in.grad, slabbed_in.grad)


def test_alignment_loss_matches_the_logit_entry_point():
    # alignment_loss is the same recursion as rnnt_loss, entered one level lower: the caller
    # supplies the alignment-grid log-probs instead of a lattice. Driving both from the same logits
    # must give the same cost, or the pruned path (which only has the lower entry) is scoring a
    # different model.
    import torch
    import torch.nn.functional as F
    from src.shared_kernel.RnntLoss import alignment_loss, rnnt_loss

    torch.manual_seed(0)
    b, t, u, v, blank = 2, 6, 3, 7, 6
    logits = torch.randn(b, t, u + 1, v, dtype=torch.float64)
    targets = torch.randint(0, blank, (b, u), dtype=torch.int32)
    frames = torch.tensor([6, 5], dtype=torch.int32)
    labels = torch.tensor([3, 2], dtype=torch.int32)

    log_probs = F.log_softmax(logits, dim=-1)
    blank_lp = log_probs[..., blank].clone()
    label_lp = (
        log_probs[:, :, :u, :]
        .gather(3, targets.long().view(b, 1, u, 1).expand(b, t, u, 1))
        .squeeze(3)
    )

    cost, alpha, beta = alignment_loss(blank_lp, label_lp, frames.long(), labels.long())
    reference = rnnt_loss(logits, targets, frames, labels, blank=blank, reduction="none")

    assert torch.allclose(cost, reference, atol=1e-9)
    assert alpha.shape == (b, t, u + 1)
    assert beta.shape == (b, t, u + 1)
    # alpha[0,0] is the empty prefix: probability 1, log 0.
    assert torch.allclose(alpha[:, 0, 0], torch.zeros(b, dtype=torch.float64), atol=1e-12)
    # beta[0,0] is the same total probability read backwards. Row 1 is ragged in BOTH axes, so this
    # only holds if alignment_loss masked the padding itself -- an unmasked beta sees a padded
    # neighbour's arcs and leaks probability in.
    assert torch.allclose(beta[:, 0, 0], -cost, atol=1e-9)
