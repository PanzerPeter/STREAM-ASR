# Pruned RNN-T: the two-stage objective that keeps the real joiner off the full alignment grid.
#
# Stage 1 (here) is a *linear* joiner, logits[b,t,u,v] = am[b,t,v] + lm[b,u,v]. Linearity is the
# whole point: because the two terms are separable, the [B, T, U+1, V] lattice never has to be a
# stored activation. forward builds it a frame-slab at a time to read off the two arcs the scan
# needs and throws it away; backward rebuilds the same slabs from `am` and `lm` and folds the
# gradient straight down to [B, T, V] and [B, U+1, V]. Nothing lattice-sized survives either call,
# so a model whose real joiner cannot afford the grid can still afford these posteriors.
#
# Those posteriors are all stage 1 is for. Their occupancy says where in the transcript the model
# thinks it is at each frame, which picks the narrow band of the grid the real (non-linear) joiner
# is then evaluated on.
import torch
import torch.nn.functional as F

from src.shared_kernel.RnntLoss import _NEG, _frame_slab, alignment_loss


def _arc_posteriors(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    log_partition: torch.Tensor,
    frames: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Posterior mass of each arc: P(paths through the arc) / P(all paths), the same quantity
    # _RnntLoss.backward computes. Arcs outside the utterance's rectangle carry a _NEG term and
    # underflow to exactly 0.
    batch, num_frames, num_states = blank_lp.shape
    num_labels = num_states - 1
    rows = torch.arange(batch, device=blank_lp.device)

    # beta one step on in t and in u, with the accepting state written in so that the last real
    # cell sees log-prob 0 instead of falling off the grid.
    beta_next_t = torch.cat([beta[:, 1:], beta.new_full((batch, 1, num_states), _NEG)], dim=1)
    beta_next_t[rows, frames - 1, labels] = 0.0
    beta_next_u = torch.cat([beta[:, :, 1:], beta.new_full((batch, num_frames, 1), _NEG)], dim=2)

    norm = log_partition.view(batch, 1, 1)
    blank_post = torch.exp(alpha + blank_lp + beta_next_t - norm)  # [B, T, U+1]
    label_post = torch.exp(
        alpha[:, :, :num_labels] + label_lp + beta_next_u[:, :, :num_labels] - norm
    )  # [B, T, U]
    return blank_post, label_post


def _emit_closure(blank_in: torch.Tensor, px_t: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    # Close the intra-frame emit chain for a whole frame at once. A path enters frame t by a blank
    # arc at some band position j, then emits s - j labels without consuming a frame, so
    #   alpha[t, s] = logsumexp_{j <= s} ( blank_in[j] + sum_{k in [j, s)} px[t, k] )
    # and with csum the exclusive cumulative sum of px, that inner sum is csum[s] - csum[j]. The
    # whole [B, S, S] term tensor is then four launches, against S sequential logaddexp steps for
    # the obvious recursion -- at S = 5 inside a T-step loop, launch count is the entire cost.
    #
    # csum differences are safe against cancellation because px is _NEG only on a SUFFIX (its _NEG
    # cells are exactly the states past the end of the transcript, and band position increases with
    # s): if csum[j] already carries a _NEG then so does every term in [j, s), and the difference is
    # a huge negative that underflows to zero probability, which is the right answer. The only case
    # where both csums carry the same _NEG mass is s == j, where the difference is identically 0.
    csum = F.pad(px_t.cumsum(dim=1), (1, 0))[:, : px_t.shape[1]]
    terms = blank_in.unsqueeze(1) + csum.unsqueeze(2) - csum.unsqueeze(1)
    return torch.logsumexp(terms.masked_fill(upper, _NEG), dim=2)


def rnnt_loss_pruned(
    logits: torch.Tensor,
    targets: torch.Tensor,
    s_begin: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    reduction: str = "sum",
) -> torch.Tensor:
    """Graves RNN-T loss restricted to an ``s_range``-wide band of the alignment grid.

    Args:
        logits: ``[B, T, S, V]`` real-joiner output on the band, where band position ``s`` is the
            absolute symbol position ``s_begin[b, t] + s``.
        targets: ``[B, U]`` label ids, zero-padded past ``target_lengths``.
        s_begin: ``int64 [B, T]`` band starts from `prune_ranges`. Must satisfy that function's
            four constraints; ``s_begin[:, 0] == 0`` in particular, since frame 0 is seeded at the
            start state.
        logit_lengths: ``[B]`` valid encoder frames per utterance.
        target_lengths: ``[B]`` valid labels per utterance.
        blank: blank index into ``V``.
        reduction: ``"none"`` -> ``[B]`` per-utterance costs; ``"sum"``/``"mean"`` -> scalar.

    Plain differentiable torch rather than a custom Function: alpha is ``[B, T, S]``, ~34 k values
    on a real batch, so letting autograd store the whole scan costs nothing.
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B, T, S, V], got {tuple(logits.shape)}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError('reduction should be one of "none", "sum", or "mean"')
    batch, num_frames, band, _ = logits.shape
    num_labels = targets.shape[1]
    device = logits.device
    frames = logit_lengths.long()
    labels = target_lengths.long()

    accum = torch.promote_types(logits.dtype, torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1, dtype=accum)

    # Absolute symbol position of every band cell, and the two arcs leaving it.
    s_axis = torch.arange(band, device=device).view(1, 1, band)
    absolute = s_begin.unsqueeze(2) + s_axis  # [B, T, S]
    # One dummy target column so the gather is always in bounds -- including an empty transcript,
    # where every band cell is past the end and the value read is masked away anyway.
    padded_targets = F.pad(targets.long(), (0, 1))
    label_ids = (
        padded_targets.unsqueeze(1)
        .expand(batch, num_frames, num_labels + 1)
        .gather(2, absolute.clamp(max=num_labels))
    )
    px = log_probs.gather(3, label_ids.unsqueeze(3)).squeeze(3)  # [B, T, S] emit arc
    py = log_probs[..., blank]  # [B, T, S] blank arc
    # A cell past the transcript's end is not a state at all; a cell AT the end is a state but has
    # no label left to emit.
    px = px.masked_fill(absolute >= labels.view(batch, 1, 1), _NEG)
    py = py.masked_fill(absolute > labels.view(batch, 1, 1), _NEG)

    # Band coordinates shift by d = s_begin[t] - s_begin[t-1] between frames, so a blank arc leaving
    # band position s of frame t lands at position s - d of frame t+1; read the other way, frame t's
    # cell s is entered from frame t-1's cell s + d.
    shift = (s_begin[:, 1:] - s_begin[:, :-1]).unsqueeze(2) + s_axis  # [B, T-1, S]
    in_band = (shift >= 0) & (shift < band)
    upper = torch.triu(torch.ones(band, band, dtype=torch.bool, device=device), diagonal=1)
    invalid = absolute > labels.view(batch, 1, 1)

    # Frame 0 is entered at the start state (0, 0) only, which s_begin[:, 0] == 0 places at band
    # position 0.
    seed = logits.new_full((batch, band), _NEG, dtype=accum)
    seed[:, 0] = 0.0
    alpha_t = _emit_closure(seed, px[:, 0], upper).masked_fill(invalid[:, 0], _NEG)
    rows_alpha = [alpha_t]
    for t in range(1, num_frames):
        leaving = alpha_t + py[:, t - 1]
        blank_in = leaving.gather(1, shift[:, t - 1].clamp(0, band - 1)).masked_fill(
            ~in_band[:, t - 1], _NEG
        )
        alpha_t = _emit_closure(blank_in, px[:, t], upper).masked_fill(invalid[:, t], _NEG)
        # Floor at _NEG so a chain of impossible cells cannot walk off toward -inf over T frames,
        # the same role the inject plane plays in RnntLoss._scan.
        alpha_t = alpha_t.clamp(min=_NEG)
        rows_alpha.append(alpha_t)
    alpha = torch.stack(rows_alpha, dim=1)  # [B, T, S]

    # The accepting state: all labels emitted, all frames consumed. Its band position is
    # labels - s_begin at the last valid frame; if s_begin violated its constraints that falls
    # outside the band, and the cost comes back at ~1e30 rather than silently reading a wrong cell.
    rows = torch.arange(batch, device=device)
    last_t = (frames - 1).clamp(min=0)
    accept_s = labels - s_begin.gather(1, last_t.view(batch, 1)).squeeze(1)
    reachable = (accept_s >= 0) & (accept_s < band)
    accept_s = accept_s.clamp(0, band - 1)
    total = alpha[rows, last_t, accept_s] + py[rows, last_t, accept_s]
    costs = -torch.where(reachable, total, total.new_full((), _NEG))

    if reduction == "sum":
        return costs.sum()
    if reduction == "mean":
        return costs.mean()
    return costs


def prune_ranges(
    occupancy: torch.Tensor,
    frames: torch.Tensor,
    labels: torch.Tensor,
    s_range: int,
) -> torch.Tensor:
    """Per-frame start of the `s_range`-wide band the real joiner is evaluated on.

    The band is centred on the expected symbol position under the simple model's occupancy, then
    forced to satisfy the four constraints a complete path needs: it must contain (0, 0), it must
    contain the accepting state, it must be monotone (time only moves forward through symbols), and
    consecutive bands must overlap so a blank arc from frame t lands inside frame t+1's band.
    Violate any one and the pruned lattice has no path at all, and the loss is -inf rather than
    merely loose.

    Returns ``int64 [B, T]``; callers use ``min(s_range, U+1)`` as the effective band width.
    """
    batch, num_frames, num_states = occupancy.shape
    device = occupancy.device
    grow = s_range - 1  # a band may advance at most this far per frame and still overlap

    # Top of the band's range: past it the accepting state (u = labels) falls out of the window.
    top = (labels + 1 - s_range).clamp(min=0).view(batch, 1)
    last = (frames - 1).clamp(min=0).view(batch, 1)
    t_axis = torch.arange(num_frames, device=device).view(1, num_frames)
    # The feasible envelope, in closed form. `ceiling` is how far a band starting at 0 can have
    # advanced by frame t; `floor` is how far it must already have advanced to still reach `top` by
    # the last valid frame. Past the last frame both collapse to `top`, which is what pins the
    # padded tail to the row's final value.
    ceiling = torch.minimum(top, t_axis * grow)
    floor = (top - (last - t_axis).clamp(min=0) * grow).clamp(min=0)

    positions = torch.arange(num_states, device=device, dtype=occupancy.dtype)
    centre = (occupancy * positions).sum(-1).round().long() - s_range // 2

    # Greedy left to right: take the centred start, then clip it into what the previous frame and
    # the envelope allow. One pass, and every constraint holds by construction -- clamping the
    # whole tensor and repairing it afterwards does not converge, because repairing the tail
    # reintroduces gaps at the head and vice versa. T is ~450 and this runs once per batch, not
    # once per symbol.
    start = occupancy.new_zeros((batch, num_frames), dtype=torch.int64)
    prev = start[:, 0]
    for t in range(1, num_frames):
        low = torch.maximum(prev, floor[:, t])
        high = torch.minimum(prev + grow, ceiling[:, t])
        prev = centre[:, t].clamp(min=low, max=high)
        start[:, t] = prev
    return start


class _RnntLossSimple(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        am: torch.Tensor,
        lm: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_frames, vocab = am.shape
        num_states = lm.shape[1]
        num_labels = num_states - 1
        frames = logit_lengths.long()
        labels = target_lengths.long()

        # promote_types rather than a hard float32 so an fp64 caller stays in fp64 and the
        # equivalence tests against the reference loss remain meaningful.
        accum = torch.promote_types(am.dtype, torch.float32)
        blank_lp = am.new_empty((batch, num_frames, num_states), dtype=accum)
        label_lp = am.new_empty((batch, num_frames, num_labels), dtype=accum)
        index = targets.long().view(batch, 1, num_labels, 1)
        slab = _frame_slab(batch, num_frames, num_states, vocab)
        for start in range(0, num_frames, slab):
            stop = min(start + slab, num_frames)
            log_probs = torch.log_softmax(
                am[:, start:stop].unsqueeze(2) + lm.unsqueeze(1), dim=-1, dtype=accum
            )
            blank_lp[:, start:stop] = log_probs[..., blank]
            label_lp[:, start:stop] = (
                log_probs[:, :, :num_labels, :]
                .gather(3, index.expand(batch, stop - start, num_labels, 1))
                .squeeze(3)
            )

        # alignment_loss masks blank_lp/label_lp in place; backward reads the masked values.
        cost, alpha, beta = alignment_loss(blank_lp, label_lp, frames, labels)

        # Occupancy is the BLANK-arc posterior, not the node posterior exp(alpha + beta - Z). A path
        # visits several u at the same t -- emit arcs do not advance the frame -- so the node
        # posterior sums to more than 1 per frame and is not a distribution over symbol positions.
        # The blank arc is taken exactly once per frame, from exactly one u, so exp(alpha + blank_lp
        # + beta[t+1] - Z) IS that distribution, which is what the band must be centred on.
        occupancy = _arc_posteriors(alpha, beta, blank_lp, label_lp, -cost, frames, labels)[
            0
        ].detach()
        outside = ~(
            (torch.arange(num_frames, device=am.device).view(1, -1, 1) < frames.view(-1, 1, 1))
            & (torch.arange(num_states, device=am.device).view(1, 1, -1) <= labels.view(-1, 1, 1))
        )
        occupancy = occupancy.masked_fill(outside, 0.0)

        ctx.save_for_backward(am, lm, targets, blank_lp, label_lp, alpha, beta, frames, labels)
        ctx.blank = blank
        ctx.mark_non_differentiable(occupancy)
        return cost, occupancy

    @staticmethod
    @torch.autograd.function.once_differentiable
    # See RnntLoss.backward: `once_differentiable` erases the named `ctx` the base declares.
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, *grad_outputs: torch.Tensor
    ):
        # `occupancy` is marked non-differentiable in forward, so its slot is only ever a
        # zero-filled placeholder.
        grad_cost, _ = grad_outputs
        am, lm, targets, blank_lp, label_lp, alpha, beta, frames, labels = ctx.saved_tensors
        batch, num_frames, vocab = am.shape
        num_states = lm.shape[1]
        num_labels = num_states - 1

        rows = torch.arange(batch, device=am.device)
        log_partition = alpha[rows, frames - 1, labels] + blank_lp[rows, frames - 1, labels]
        blank_post, label_post = _arc_posteriors(
            alpha, beta, blank_lp, label_lp, log_partition, frames, labels
        )

        # d(-log Z)/d logits[v] = softmax(logits)[v] * sum_k dZ/d log_probs[k] - dZ/d log_probs[v],
        # and only the blank and the utterance's own next label have a non-zero dZ/d log_probs.
        upstream = grad_cost.view(batch, 1, 1)
        scale = (blank_post + F.pad(label_post, (0, 1))) * upstream
        blank_post = blank_post * upstream
        label_post = -label_post * upstream

        # The lattice gradient is summed away along u (for am) and along t (for lm) inside the slab
        # loop, so the [B, T, U+1, V] tensor is a transient here exactly as it was in forward.
        grad_am = am.new_zeros((batch, num_frames, vocab), dtype=alpha.dtype)
        grad_lm = lm.new_zeros((batch, num_states, vocab), dtype=alpha.dtype)
        index = targets.long().view(batch, 1, num_labels, 1)
        slab = _frame_slab(batch, num_frames, num_states, vocab)
        for start in range(0, num_frames, slab):
            stop = min(start + slab, num_frames)
            part = torch.softmax(
                am[:, start:stop].unsqueeze(2) + lm.unsqueeze(1), dim=-1, dtype=alpha.dtype
            )
            part.mul_(scale[:, start:stop].unsqueeze(3))
            part[..., ctx.blank] -= blank_post[:, start:stop]
            part[:, :, :num_labels, :].scatter_add_(
                3,
                index.expand(batch, stop - start, num_labels, 1),
                label_post[:, start:stop].unsqueeze(3),
            )
            grad_am[:, start:stop] = part.sum(dim=2)
            grad_lm += part.sum(dim=1)
        return grad_am.to(am.dtype), grad_lm.to(lm.dtype), None, None, None, None


def rnnt_loss_simple(
    am: torch.Tensor,
    lm: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RNN-T loss over the linear joiner ``am[b,t,v] + lm[b,u,v]``, plus its state occupancy.

    Args:
        am: ``[B, T, V]`` encoder-side projection to vocabulary width.
        lm: ``[B, U+1, V]`` predictor-side projection to vocabulary width.
        targets: ``[B, U]`` label ids, zero-padded past ``target_lengths``.
        logit_lengths: ``[B]`` valid encoder frames per utterance.
        target_lengths: ``[B]`` valid labels per utterance.
        blank: blank index into ``V``.

    Returns:
        ``(cost [B], occupancy [B, T, U+1])``. ``cost`` is the per-utterance ``-log P``, matching
        ``rnnt_loss(..., reduction="none")`` on the same lattice built explicitly. ``occupancy`` is
        the per-frame distribution over symbol positions (the blank-arc posterior; it sums to 1
        across ``u`` within every valid frame), detached and zeroed outside each utterance's
        rectangle -- it feeds `prune_ranges`, which turns it into integer band starts, so it
        carries no gradient.
    """
    if am.ndim != 3:
        raise ValueError(f"am must be [B, T, V], got {tuple(am.shape)}")
    if lm.ndim != 3:
        raise ValueError(f"lm must be [B, U+1, V], got {tuple(lm.shape)}")
    cost, occupancy = _RnntLossSimple.apply(am, lm, targets, logit_lengths, target_lengths, blank)
    return cost, occupancy
