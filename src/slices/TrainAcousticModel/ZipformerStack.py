from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.MaskUtils import make_chunk_mask, make_pad_mask
from src.slices.TrainAcousticModel.Resample import SimpleDownsample, SimpleUpsample
from src.slices.TrainAcousticModel.StreamCache import AttnCache, ConvCache
from src.shared_kernel.BiasNorm import BiasNorm
from src.slices.TrainAcousticModel.ZipformerBlock import ZipformerBlock

# Power iteration converges at (sigma2/sigma1)^k, so the cost is set by how flat the top of the
# spectrum is, and a FRESH matrix is the flat case: sigma1/sigma2 is 1.007-1.042 over the five
# in_proj at init against 1.20-1.47 once trained. MEASURED against
# `torch.linalg.matrix_norm(ord=2)`, cold from a random vector on the flattest of them (384x256):
# 0.99869 of the true value at 100 iterations, 0.99980 at 200, 1.00000 at 400. Warm-started and
# re-run after each of 30 spectrally-flat perturbations of 0.4 % of the weight norm -- the rate
# the measured run drifted its trunk at -- two iterations never fell below 0.99999 on any of them.
#
# So the cold count buys the accuracy once and the warm one keeps it for ~nothing. It matters only
# in the direction of a slight UNDER-estimate, which leaves the enforced bound a hair loose; at
# init, where the estimate is worst, sigma is ~1.2 against a bound of 10 and nothing is projected
# at all.
_POWER_ITERS = 2
_POWER_ITERS_COLD = 400
# How far over the bound `project` still trusts a rank-1 deflation to be the whole projection.
# In training the excess is a fraction of a percent -- the weight drifts ~0.4 % of its norm per
# optimizer step and the projection runs after every one of them -- so this never routes to the
# SVD there. It exists for a weight arriving from outside that loop (a checkpoint saved under a
# looser bound, or none), where many singular values can sit above the ceiling at once and
# deflating the top one would only promote the second.
_SVD_FALLBACK_RATIO = 1.05


class ZipformerStack(nn.Module):
    """One Zipformer stack. Projects to the stack width, optionally downsamples to a lower
    frame rate, runs the blocks there, upsamples back to the base rate, and mixes with the
    stack input through a learnable scalar bypass. Output frame count == input frame count.

    `in_proj` is the model's TRUNK: the blocks are all pre-normed and the last one exits through a
    `BiasNorm`, so the processed half of the mix is bounded, but the residual half passes through
    this Linear and no normaliser at all. Both halves are bounded here -- see `project`.
    """

    sigma_v: torch.Tensor
    sigma_u: torch.Tensor

    def __init__(
        self, dim_in: int, dim: int, num_layers: int, downsample: int, num_heads: int
    ) -> None:
        super().__init__()
        model = get_config().model
        self.in_proj = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()
        # THE TRUNK NORMALISER. `in_proj` is the only operator between two stacks and every other
        # normaliser in the encoder sits on a branch, so until this existed the residual stream's
        # amplitude was bounded only through the parameters that produce it -- and six successive
        # attempts to do that each moved the inflation into whatever summary of the matrix was left
        # free (CLAUDE.md pitfalls 8, 12, 13, 14, 16). This bounds the ACTIVATION, which is the
        # quantity that failed every time. Stack 0's `in_proj` is an `Identity` and there is no
        # preceding stack to compound from, so it is left alone; that entry of `stack_mix/*_trunk`
        # is the frontend's output.
        self.trunk_norm: nn.Module = (
            BiasNorm(dim, log_scale_max=model.trunk_norm_log_scale_max)
            if model.trunk_norm and dim_in != dim
            else nn.Identity()
        )
        self.downsample = SimpleDownsample(downsample) if downsample > 1 else None
        self.upsample = SimpleUpsample(downsample) if downsample > 1 else None
        # Value residual is stack-local: block-0 produces the values injected into every deeper
        # block, so the shortcut runs at this stack's width and frame rate. The gate is a learnable
        # scalar per deeper block (init from config); block-0 never receives a residual so it is 0.
        lam = model.encoder_value_residual_lambda
        self.blocks = nn.ModuleList(
            [
                ZipformerBlock(dim, num_heads, value_residual_init=0.0 if i == 0 else lam)
                for i in range(num_layers)
            ]
        )
        self.bypass = nn.Parameter(torch.tensor(0.5))  # residual↔processed interpolation
        self.bypass_min = model.stack_bypass_min
        self.in_proj_max_sigma = model.stack_in_proj_max_sigma
        if isinstance(self.in_proj, nn.Linear):
            # Singular-vector iterates for the spectral projection, warm-started from the previous
            # step. NOT persisted: they are a cache of a quantity derivable from the weight, and
            # `_POWER_ITERS_COLD` re-converges them on the first call after any load. `sigma_u` is
            # the MATCHED left vector -- `W @ sigma_v == sigma * sigma_u` -- which is what makes the
            # rank-1 deflation in `project` exact rather than approximate.
            self.register_buffer(
                "sigma_v", F.normalize(torch.randn(dim_in), dim=0), persistent=False
            )
            self.register_buffer("sigma_u", torch.zeros(dim), persistent=False)
            self._sigma_warm = False
        # Realized trunk amplitude, written by `forward` for logging. THE quantity every bound in
        # this file is a proxy for, and the one no metric read through four collapses: a parameter
        # norm bounds the gain against the worst-case input direction, this is the gain the data
        # actually gets. Stack 0's `in_proj` is an Identity, so its value is the frontend's output
        # amplitude -- the one trunk operator no bound reaches.
        self.trunk_rms: torch.Tensor | None = None

    @torch.no_grad()
    def trunk_sigma(self) -> torch.Tensor:
        """Largest singular value of `in_proj`, by power iteration warm-started across steps.

        This is the gain the trunk applies to its worst-case input direction, and therefore an
        upper bound on the gain it applies to the data -- which is the whole reason it replaced
        `||W||_F / sqrt(n_out)`. That quantity is the gain against ISOTROPIC input and the
        inflation is not isotropic: MEASURED 2026-08-22 at step 43,200 of the 600k run,
        `stacks.3.in_proj` read 2.76 isotropic against a realized 6.5x on real dev audio
        (trunk RMS 43.19 out of a stack-2 output of 6.61), and sigma1/isotropic ran 2.1-3.7 across
        the five stacks and rose at every checkpoint. The isotropic bound of 4.0 clipped nothing
        in 43k steps while the realized trunk amplitude grew 7x.

        Two iterations per step because the weight moves by ~0.4 % of its norm per step, so the
        iterate is already converged from the previous call; `_POWER_ITERS_COLD` covers the first
        call, when `sigma_v` is still the random init or has just come back non-persistent from a
        checkpoint load.

        The left vector is recomputed from the FINAL `v` rather than carried out of the loop one
        half-step stale, so `sigma_u`/`sigma_v` are a matched pair with `W @ v == sigma * u`. That
        costs one extra matvec on a 512x384 and it is what `project`'s deflation needs to hit the
        bound exactly instead of overshooting by the staleness.
        """
        weight = cast(nn.Linear, self.in_proj).weight
        iters = _POWER_ITERS if self._sigma_warm else _POWER_ITERS_COLD
        self._sigma_warm = True
        v = self.sigma_v
        for _ in range(iters):
            u = F.normalize(weight @ v, dim=0, eps=1e-12)
            v = F.normalize(weight.t() @ u, dim=0, eps=1e-12)
        wv = weight @ v
        sigma = torch.linalg.vector_norm(wv)
        self.sigma_v.copy_(v)
        self.sigma_u.copy_(wv / sigma.clamp_min(1e-12))
        return sigma

    @torch.no_grad()
    def trunk_stable_rank(self) -> torch.Tensor:
        """`||W||_F^2 / sigma_1^2` for `in_proj` -- how many directions the trunk still carries.

        THE metric the 2026-08-24 collapse needed and no one logged. A bound on `sigma_1` says
        nothing about the rest of the spectrum, and the uniform-rescale projection this file used
        until then removed the excess from every direction at once, so with the bound binding
        continuously the top direction was held at the ceiling while everything else decayed
        geometrically. MEASURED on `stacks.3.in_proj` of the 600k run, with `sigma_1` pinned at
        exactly 10.00 the whole time: 64.4 at the BEST-RQ warm start, 48.9 at step 75.6k, 32.6 at
        86.4k, 21.7 at 97.2k -- halving every ~20k steps, on the same clock as the realized trunk
        RMS doubled (2.02 -> 12.34 against 1.51 for the model that shipped 3.43 % test-clean).

        Cheap: the Frobenius norm is one pass over the weight and `sigma_1` comes warm out of
        `trunk_sigma`. Falling here means the trunk is turning into a rank-1 amplifier, which is
        both an amplitude failure (the input rotates into the one surviving direction, so the
        realized gain climbs toward `sigma_1` with `sigma_1` frozen) and a capacity failure (the
        stack downstream receives a rank-deficient input).
        """
        weight = cast(nn.Linear, self.in_proj).weight
        sigma = self.trunk_sigma()
        return weight.pow(2).sum() / sigma.clamp_min(1e-12).pow(2)

    @torch.no_grad()
    def project(self) -> None:
        """Bound both halves of the mix. Trainers call this after every optimizer step.

        THE GATE, into `[model.stack_bypass_min, 1]`. `clamp` has a hard dead zone: its gradient is
        exactly 0 past the bounds (1.0 AT a bound, 0.0 beyond), so a gate the optimizer pushes
        outside stops receiving gradient permanently and can only crawl back through weight decay.
        MEASURED 2026-08-05: `encoder.stacks.5.bypass` sat at 1.0020-1.0049 across the whole
        394k-416k window with AdamW `exp_avg` at -2.3e-22 -- frozen out of training, oscillating
        over the boundary as weight decay pulled it under and one live step shoved it back over.
        Projecting after the update is ordinary projected gradient descent, and it leaves the gate
        resting exactly ON the bound where gradient still flows, so it can always come back down.

        The floor is not zero because zero is an ABSORBING state, not just a dead zone: at b = 0
        the stack is `out = in_proj(input)` and everything inside its blocks is multiplied by 0, so
        those parameters receive exactly no gradient and stop being part of the model. MEASURED on
        transducer_step81000.pt, stack 2 at b = 0.0: 0 of its 105 block parameters had a nonzero
        gradient, 10.6 M of 53.8 M encoder parameters frozen, with `d(loss)/d(bypass)` = +0.91 --
        descent pressing it further into the floor. See config/model.yaml.

        THE TRUNK, to `sigma_max(W) <= model.stack_in_proj_max_sigma`. `in_proj` is the one
        operator between stacks, so the encoder's amplitude is the product of these gains, and
        nothing else in the model bounds it: Muon's Newton-Schulz sets every singular value of the
        update to 1, so this direction takes a full-size step on a radial gradient of 2.5e-5, and
        weight decay's equilibrium is 175x the init. Left free it put RMS 8,193 into stack 3 and
        reduced its first block to an identity (1 - cos(in, out) = 0.03).

        Clipping ONLY the singular values above the bound, by deflating the top one:
        `W -= (sigma_1 - c) * u_1 v_1^T`. That is the minimal (Frobenius-nearest) projection onto
        the ball, and both vectors are already in hand from `trunk_sigma`, so it costs one rank-1
        update rather than the SVD per step it looks like it needs.

        THIS USED TO BE A UNIFORM RESCALE, `W *= c / sigma_1`, and that is the 2026-08-24 failure.
        A rescale trims every direction by the same factor while the gradient re-inflates only the
        top one, so it is harmless exactly as long as it fires rarely -- which is what the old
        version of this docstring assumed, and what stopped being true. MEASURED on the 600k run:
        `trunk_gain_max` reached the ceiling at step 71,000 and bound on 99.2 % of every logged
        step after it, so the non-top spectrum was multiplied down ~2,000 times per e-fold of drift
        and `stacks.3.in_proj`'s stable rank went 64.4 (warm start) -> 48.9 (75.6k) -> 32.6 (86.4k)
        -> 21.7 (97.2k) with `sigma_1` reading exactly 10.00 throughout. The realized trunk RMS
        doubled on the same clock -- 4.45 (70k) -> 12.34 (102k) against 1.51 for the model that
        shipped 3.43 % test-clean -- because a rank-collapsing operator drags its own input into
        the one direction it has left, so the gain the data gets climbs toward the worst case while
        the worst case itself sits frozen on the bound. The bound became the mechanism.

        Deflating the top direction is the whole projection only when `sigma_2 <= c`, and per-step
        projection maintains that: the weight drifts ~0.4 % of its norm per step, so nothing else
        can cross the bound between two calls, and a trained `in_proj` is anisotropic enough that
        it is not close (sigma_2/sigma_1 was 0.72-0.79 across the run's stacks). Several values
        over the bound at once is reachable only by LOADING a weight saved without this constraint,
        where the spectrum can be flat -- a fresh matrix is the flat case, sigma_1/sigma_2 is
        1.007-1.042 at init -- and one deflation there would trade the top value for the second.
        `_SVD_FALLBACK_RATIO` routes that case to an exact `min(sigma_i, c)` clip: it pays an SVD,
        but only on a weight that is grossly out of range, never in the loop.

        `train/trunk_stable_rank_min` is what makes a flat top visible if one ever develops in
        training -- see `trunk_stable_rank`.

        `in_proj.bias` is an offset rather than a gain and is left alone.

        `forward` still clamps the gate it uses, so a checkpoint carrying an out-of-range one
        computes correctly from the first batch, before the first projection lands. Reached from
        `shared_kernel.ParameterProjection.project_constraints`; see that module for why the
        projection is a separate call and not part of `forward`.
        """
        self.bypass.clamp_(self.bypass_min, 1.0)
        if not isinstance(self.in_proj, nn.Linear):
            return
        bound = self.in_proj_max_sigma
        sigma = self.trunk_sigma()
        if sigma <= bound:
            return
        weight = self.in_proj.weight
        if sigma > bound * _SVD_FALLBACK_RATIO:
            u, s_vals, vh = torch.linalg.svd(weight, full_matrices=False)
            weight.copy_(u @ torch.diag(s_vals.clamp(max=bound)) @ vh)
            # Every singular direction moved, so the warm iterate is meaningless now.
            self._sigma_warm = False
            return
        weight.sub_(torch.outer(self.sigma_u * (sigma - bound), self.sigma_v))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        base_pad_mask: torch.Tensor,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        x = self.trunk_norm(self.in_proj(x))
        residual = x
        # POST-norm, i.e. what the residual stream actually carries into the blocks and into the
        # next stack. That is the quantity the amplitude ledger is about; with `trunk_norm` on it
        # is bounded by construction and this metric confirms the bound rather than leading a
        # divergence, which is the point of having it.
        #
        # fp32 accumulation without materialising a copy of the whole activation; padded frames
        # carry `in_proj.bias` rather than 0 and are counted, which at ~0.2 % frame padding moves
        # this by less than the quantity's step-to-step noise.
        self.trunk_rms = (
            torch.linalg.vector_norm(residual.detach(), dtype=torch.float32)
            / residual.numel() ** 0.5
        )
        base_len = x.shape[1]

        if self.downsample is not None:
            x, ds_lengths = self.downsample(x, lengths)
            pad_mask = make_pad_mask(ds_lengths, x.shape[1])
            # Chunk size is expressed in base-rate frames; scale to this stack's downsampled rate.
            local_chunk = max(1, chunk_size // self.downsample.factor) if chunk_size > 0 else 0
        else:
            pad_mask = base_pad_mask
            local_chunk = chunk_size

        attn_visible = (
            make_chunk_mask(x.shape[1], local_chunk, x.device) if chunk_size > 0 else None
        )
        v0: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            x, v = block(x, pad_mask, attn_visible, value_residual=None if i == 0 else v0)
            if i == 0:
                v0 = v

        if self.upsample is not None:
            x = self.upsample(x, out_len=base_len)

        bypass = self.bypass.clamp(self.bypass_min, 1.0)
        return residual + bypass * (x - residual)

    def streaming_forward(
        self,
        x: torch.Tensor,
        attn_caches: list[AttnCache],
        conv_caches: list[ConvCache],
    ) -> tuple[torch.Tensor, list[AttnCache], list[ConvCache]]:
        # Chunks are aligned to downsample factor, so downsample never straddles boundary.
        # Bypass and up/down-sample are per-frame linear ops. Streaming reduces to running
        # each block (causal conv + KV-cached attn) at the stack's rate.
        # Per-frame over channels, so it adds no cross-chunk dependency and streaming stays
        # exactly equivalent to the batched path (test_streaming_forward_equivalence).
        x = self.trunk_norm(self.in_proj(x))
        residual = x
        base_len: int = x.shape[1]
        if self.downsample is not None:
            frame_len: int = x.shape[1]
            lengths: torch.Tensor = torch.tensor([frame_len], device=x.device)
            x, _ = self.downsample(x, lengths)
        new_ac: list[AttnCache] = []
        new_cc: list[ConvCache] = []
        v0: torch.Tensor | None = None
        for i, (ac, cc) in enumerate(zip(attn_caches, conv_caches)):
            block = cast(ZipformerBlock, self.blocks[i])
            x, v, ac_new, cc_new = block.streaming_forward(
                x, ac, cc, value_residual=None if i == 0 else v0
            )
            if i == 0:
                v0 = v
            new_ac.append(ac_new)
            new_cc.append(cc_new)
        if self.upsample is not None:
            x = self.upsample(x, out_len=base_len)
        bypass = self.bypass.clamp(self.bypass_min, 1.0)
        return residual + bypass * (x - residual), new_ac, new_cc
