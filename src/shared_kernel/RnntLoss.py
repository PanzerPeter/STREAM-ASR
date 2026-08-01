# RNN-T (Graves) loss: pure-PyTorch forward-backward.
#
# Replaces torchaudio.transforms.RNNTLoss, which on this build spends ~191 ms on the
# [14, 415, 118, 501] lattice a training batch produces -- ~50x off the card's memory roofline, and
# the single largest cost in a transducer step. The math here is the same recursion and the same
# analytic gradient, so checkpoints are interchangeable between the two; only the schedule changes:
#
#   * The two unavoidable [B, T, U+1, V] passes (log-softmax, gradient) become single fused
#     elementwise kernels instead of being interleaved into the alpha/beta scan.
#   * The scan runs on the [B, T, U+1] *alignment* grid -- 500x smaller than the lattice -- as a
#     loop over anti-diagonals t+u=d. Cells on one anti-diagonal depend only on the previous one,
#     so each d is one vectorised step and the recursion is T+U steps rather than T*U. Both the
#     forward and the backward variable advance in the SAME loop (see _scan): at ~1e3 numbers per
#     step the recursion is pure launch overhead, so the step count is what costs.
#   * Nothing lattice-sized survives the forward call: the fp32 log-probabilities are freed when
#     forward returns and the softmax is recomputed in backward, which costs one extra elementwise
#     pass and removes a [B, T, U+1, V] fp32 tensor from the step's peak.
#
# Anti-diagonals are addressed through a *sheared* view, shear[d, u] = grid[d - u, u], which turns
# "the cells with t+u=d" into "row d" so a step is a vector op rather than a strided gather.
import torch
import torch.nn.functional as F

# Log-domain "impossible". Finite rather than -inf so a masked cell surviving a few hundred
# logaddexp steps can never produce inf - inf = NaN, and floored back to _NEG each step (see the
# inject plane in _scan) so a chain of NEG + NEG cannot walk off toward -inf either. exp(_NEG)
# underflows to exactly 0, which is the gradient an unreachable alignment must contribute.
_NEG = -1.0e30

# Ceiling on the fp32 working set of the two lattice-sized passes. The bf16 lattice itself has to
# exist -- the joiner produced it -- but its fp32 log-softmax is a pure transient, so both passes
# run a frame-slab at a time. That caps the extra allocation here instead of letting it reach
# 4*B*T*(U+1)*V, ~1 GiB on a dense LibriSpeech bucket, which is the difference between fitting the
# default frame budget on a 12 GB card and taking an OOM on the densest few batches per epoch.
_WORK_BYTES = 128 * 1024 * 1024


def _frame_slab(batch: int, num_frames: int, num_states: int, vocab: int) -> int:
    # Frames per slab such that one fp32 slab stays under _WORK_BYTES (at least one frame).
    per_frame = batch * num_states * vocab * 4
    return max(1, min(num_frames, _WORK_BYTES // max(per_frame, 1)))


def _shear_map(
    num_frames: int, num_states: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (t, u) coordinates of every anti-diagonal cell plus a validity mask for the (d, u) pairs
    # that fall off the grid. num_diagonals = T + U covers t+u for every real cell; the accepting
    # state at (T, U) sits one past the end and is handled by the caller's tail seed.
    num_diagonals = num_frames + num_states - 1
    d = torch.arange(num_diagonals, device=device).unsqueeze(1)
    u = torch.arange(num_states, device=device).unsqueeze(0)
    t = d - u
    valid = (t >= 0) & (t < num_frames)
    return (
        t.clamp_(0, num_frames - 1).expand(num_diagonals, num_states),
        u.expand(num_diagonals, num_states),
        valid,
    )


def _shear(
    grid: torch.Tensor, t_idx: torch.Tensor, u_idx: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    # [B, T, U+1] -> [B, D, U+1], off-grid cells set to _NEG.
    return torch.where(valid, grid[:, t_idx, u_idx], grid.new_full((), _NEG))


def _unshear_map(
    num_frames: int, num_states: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Inverse of the shear: grid[t, u] = shear[t + u, u]. Every (t, u) lies on exactly one
    # anti-diagonal, so this is a total map -- no mask, and therefore no boolean-index
    # `nonzero` and no device->host sync on the hot path.
    t = torch.arange(num_frames, device=device).unsqueeze(1)
    u = torch.arange(num_states, device=device).unsqueeze(0)
    return (t + u).expand(num_frames, num_states), u.expand(num_frames, num_states)


def _unshear(sheared: torch.Tensor, d_idx: torch.Tensor, u_idx: torch.Tensor) -> torch.Tensor:
    # [B, D, U+1] -> [B, T, U+1].
    return sheared[:, d_idx, u_idx]


def _scan(
    blank_lp: torch.Tensor,
    label_lp: torch.Tensor,
    t_idx: torch.Tensor,
    u_idx: torch.Tensor,
    valid: torch.Tensor,
    frames: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both Graves variables, sheared, in one pass over the anti-diagonals.

    alpha[t, u] = log P(consume frames < t, emit the first u labels)
        alpha[0, 0] = 0;  alpha[t, u] = logaddexp(alpha[t-1, u] + blank_lp[t-1, u],
                                                  alpha[t,   u-1] + label_lp[t,   u-1])
    beta[t, u]  = log P(emit the remaining labels from (t, u) on)
        beta[t, u]  = logaddexp(blank_lp[t, u] + beta[t+1, u],
                                label_lp[t, u] + beta[t,   u+1])
    closed by a virtual accepting state of log-prob 0 at (logit_length, target_length). Seeding that
    state rather than special-casing the last cell is what makes beta[T-1, U] fall out as
    blank_lp[T-1, U]. Unlike alpha, beta is read at (0, 0) and therefore sees the whole grid, so its
    inputs must already be masked to each utterance's own rectangle or a padded neighbour leaks
    probability in.

    Reversed in both axes, beta obeys the same recursion as alpha -- `out[d]` from `out[d-1]` at
    the same state and at the state below -- differing only in which plane the two arc log-probs
    are read from. That is what lets one loop advance both: the two variables stack along the batch
    dimension and share every kernel launch. One diagonal touches B*(U+1) values, ~1e3 numbers, so
    this recursion is bound by launch count and nothing else -- halving the step count and cutting
    the ops per step from five to two is the entire optimisation.
    """
    batch, _, num_states = blank_lp.shape
    num_diagonals = t_idx.shape[0]
    device = blank_lp.device

    # alpha reads its blank arc from diagonal d-1 (a frame was consumed to get here) and its label
    # arc from diagonal d; beta reads both at d. Pre-shifting alpha's blank plane by one diagonal
    # makes the two index patterns identical.
    blank_s = _shear(blank_lp, t_idx, u_idx, valid)
    stay = torch.cat([F.pad(blank_s[:, :-1], (0, 0, 1, 0), value=_NEG), blank_s.flip((1, 2))])
    emit = torch.cat(
        [
            _shear(F.pad(label_lp, (1, 0), value=_NEG), t_idx, u_idx, valid),
            _shear(F.pad(label_lp, (0, 1), value=_NEG), t_idx, u_idx, valid).flip((1, 2)),
        ]
    )

    # The accepting state enters as a logaddexp against a plane that is _NEG everywhere else. That
    # doubles as the floor the recursion needs: logaddexp(x, _NEG) returns x untouched when
    # x >> _NEG and returns _NEG when x < _NEG, so it is exactly the clamp it replaces, for free.
    rows = torch.arange(batch, device=device)
    accept_diag = frames + labels
    accept_d = num_diagonals - 1 - accept_diag  # reversed-axis diagonal of the accepting state
    accept_u = num_states - 1 - labels
    inject = blank_lp.new_full((2 * batch, num_diagonals, num_states), _NEG)
    inject[batch + rows, accept_d.clamp(min=0), accept_u] = torch.where(
        accept_d >= 0, blank_lp.new_zeros(()), blank_lp.new_full((), _NEG)
    )

    # Column 0 is a permanent _NEG sentinel, so "the previous diagonal shifted one state down" --
    # the emit arc's source -- is a slice of the same row rather than a concatenate.
    buf = blank_lp.new_full((2 * batch, num_diagonals, num_states + 1), _NEG)
    buf[:batch, 0, 1] = 0.0  # alpha[0, 0]
    # An utterance that runs the full padded length puts its accepting state one diagonal past the
    # array, so for beta that state is the pre-state of diagonal 0 rather than an injection into it.
    tail = blank_lp.new_full((batch, num_states + 1), _NEG)
    tail[rows, accept_u + 1] = torch.where(
        accept_diag == num_diagonals, blank_lp.new_zeros(()), blank_lp.new_full((), _NEG)
    )
    head = buf[batch:, 0, 1:]
    torch.logaddexp(tail[:, 1:] + stay[batch:, 0], tail[:, :-1] + emit[batch:, 0], out=head)
    torch.logaddexp(head, inject[batch:, 0], out=head)

    for d in range(1, num_diagonals):
        prev, cur = buf[:, d - 1], buf[:, d, 1:]
        torch.logaddexp(prev[:, 1:] + stay[:, d], prev[:, :-1] + emit[:, d], out=cur)
        torch.logaddexp(cur, inject[:, d], out=cur)
    return buf[:batch, :, 1:], buf[batch:, :, 1:].flip((1, 2))


class _RnntLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int,
    ) -> torch.Tensor:
        batch, num_frames, num_states, vocab = logits.shape
        num_labels = num_states - 1
        device = logits.device
        frames = logit_lengths.long()
        labels = target_lengths.long()

        # Promote inside the kernel: under bf16 autocast no fp32 copy of the whole lattice is ever
        # materialised, only one slab at a time. promote_types (rather than a hard float32) keeps an
        # fp64 caller in fp64, which is what makes gradcheck meaningful.
        accum = torch.promote_types(logits.dtype, torch.float32)
        blank_lp = logits.new_empty((batch, num_frames, num_states), dtype=accum)
        label_lp = logits.new_empty((batch, num_frames, num_labels), dtype=accum)
        slab = _frame_slab(batch, num_frames, num_states, vocab)
        for start in range(0, num_frames, slab):
            stop = min(start + slab, num_frames)
            log_probs = torch.log_softmax(logits[:, start:stop], dim=-1, dtype=accum)
            blank_lp[:, start:stop] = log_probs[..., blank]
            label_lp[:, start:stop] = (
                log_probs[:, :, :num_labels, :]
                .gather(
                    3,
                    targets.long()
                    .view(batch, 1, num_labels, 1)
                    .expand(batch, stop - start, num_labels, 1),
                )
                .squeeze(3)
            )

        # Mask both to the utterance's own T x U rectangle. Required for beta, which is read at
        # (0, 0) and so sees the whole padded grid; also required for the arc posteriors in
        # backward, where an unmasked label log-prob at t == logit_length would otherwise pair with
        # the virtual accepting state and manufacture gradient on a padded frame.
        outside = ~(
            (torch.arange(num_frames, device=device).view(1, -1, 1) < frames.view(-1, 1, 1))
            & (torch.arange(num_states, device=device).view(1, 1, -1) <= labels.view(-1, 1, 1))
        )
        blank_lp.masked_fill_(outside, _NEG)
        label_lp.masked_fill_(outside[:, :, :num_labels], _NEG)

        t_idx, u_idx, valid = _shear_map(num_frames, num_states, device)
        back_d, back_u = _unshear_map(num_frames, num_states, device)
        alpha_s, beta_s = _scan(blank_lp, label_lp, t_idx, u_idx, valid, frames, labels)
        alpha = _unshear(alpha_s, back_d, back_u)
        beta = _unshear(beta_s, back_d, back_u)
        # Total log-probability read off the last real cell, matching Graves' definition:
        # log Z = alpha[T-1, U] + log P(blank | T-1, U). beta[0, 0] equals it (locked by test).
        rows = torch.arange(batch, device=device)
        log_partition = alpha[rows, frames - 1, labels] + blank_lp[rows, frames - 1, labels]

        ctx.save_for_backward(
            logits, targets, blank_lp, label_lp, alpha, beta, frames, labels, log_partition
        )
        ctx.blank = blank
        return -log_partition

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            logits,
            targets,
            blank_lp,
            label_lp,
            alpha,
            beta,
            frames,
            labels,
            log_partition,
        ) = ctx.saved_tensors
        batch, num_frames, num_states, _ = logits.shape
        num_labels = num_states - 1
        device = logits.device
        rows = torch.arange(batch, device=device)

        # beta one step on in t and in u, with the accepting state written in so that the last real
        # cell sees log-prob 0 instead of falling off the grid.
        beta_next_t = torch.cat([beta[:, 1:], beta.new_full((batch, 1, num_states), _NEG)], dim=1)
        beta_next_t[rows, frames - 1, labels] = 0.0
        beta_next_u = torch.cat(
            [beta[:, :, 1:], beta.new_full((batch, num_frames, 1), _NEG)], dim=2
        )

        # Posterior mass of each arc: P(paths through the arc) / P(all paths). Arcs outside the
        # utterance's rectangle carry a _NEG term and underflow to exactly 0.
        norm = log_partition.view(batch, 1, 1)
        blank_grad = torch.exp(alpha + blank_lp + beta_next_t - norm)  # [B, T, U+1]
        label_grad = torch.exp(
            alpha[:, :, :num_labels] + label_lp + beta_next_u[:, :, :num_labels] - norm
        )  # [B, T, U]

        # d(-log Z)/d logits[v] = softmax(logits)[v] * sum_k dZ/d log_probs[k] - dZ/d log_probs[v],
        # and only the blank and the utterance's own next label have a non-zero dZ/d log_probs.
        upstream = grad_output.view(batch, 1, 1)
        scale = (blank_grad + F.pad(label_grad, (0, 1))) * upstream
        blank_grad = blank_grad * upstream
        label_grad = -label_grad * upstream
        # Written slab-by-slab straight into a gradient of the caller's dtype, so the fp32 softmax
        # never exists at full lattice size (see _WORK_BYTES).
        grad = torch.empty_like(logits)
        vocab = logits.shape[3]
        slab = _frame_slab(batch, num_frames, num_states, vocab)
        for start in range(0, num_frames, slab):
            stop = min(start + slab, num_frames)
            part = torch.softmax(logits[:, start:stop], dim=-1, dtype=alpha.dtype)
            part.mul_(scale[:, start:stop].unsqueeze(3))
            part[..., ctx.blank] -= blank_grad[:, start:stop]
            part[:, :, :num_labels, :].scatter_add_(
                3,
                targets.long()
                .view(batch, 1, num_labels, 1)
                .expand(batch, stop - start, num_labels, 1),
                label_grad[:, start:stop].unsqueeze(3),
            )
            grad[:, start:stop] = part
        return grad, None, None, None, None


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    reduction: str = "sum",
) -> torch.Tensor:
    """Graves RNN-T loss over a joiner lattice.

    Args:
        logits: ``[B, T, U+1, V]`` joiner output, any float dtype. The log-softmax and the
            forward-backward accumulate at fp32 or the input's own precision, whichever is wider
            (so an fp64 caller stays in fp64); the returned gradient is cast back to the input.
        targets: ``[B, U]`` label ids, zero-padded past ``target_lengths``.
        logit_lengths: ``[B]`` valid encoder frames per utterance.
        target_lengths: ``[B]`` valid labels per utterance.
        blank: blank index into ``V``.
        reduction: ``"none"`` -> ``[B]`` per-utterance costs; ``"sum"``/``"mean"`` -> scalar.
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B, T, U+1, V], got {tuple(logits.shape)}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError('reduction should be one of "none", "sum", or "mean"')
    costs = _RnntLoss.apply(logits, targets, logit_lengths, target_lengths, blank)
    if reduction == "sum":
        return costs.sum()
    if reduction == "mean":
        return costs.mean()
    return costs
