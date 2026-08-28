# Trainer helpers specific to the acoustic-model slice: selective compilation, runaway detection,
# seeding, ETA formatting and activation checkpointing.
import random
import statistics
from collections import deque
from typing import cast

import torch
import torch.utils.checkpoint
import torch._dynamo

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn
from src.slices.TrainAcousticModel.ConvModule import ConvModule
from src.slices.TrainAcousticModel.TransducerJoiner import TransducerJoiner
from src.slices.TrainAcousticModel.ZipformerBlock import ZipformerBlock
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack

# The modules worth handing to inductor, in descending order of what each one measured on this
# model (RTX 5070, torch 2.11+cu128, B=20, profile_transducer_step.py, cumulative):
#
#   BiasNorm            -5.3 %    96 calls/forward; three full-size passes fuse into one
#   TransducerJoiner    -3.7 %    fuses the [B,T,U+1,J] broadcast-add + tanh feeding the readout
#   ConvModule          -1.5 %    glu -> mask -> norm -> silu chain
#   SwiGluFfn           -1.0 %    silu(gate) * up
#                      -11.5 % together, and peak VRAM 6.41 -> 5.44 GiB
#
# These are LEAF modules: pointwise/reduction chains with static control flow and no `get_config()`
# call inside. That is the whole reason this works where whole-model compilation did not. Compiling
# `ZipformerBlock` instead measured the same -11.5 % but took 135 s of warmup against 8 s and
# produced 49 graphs against 16, because `RotaryAttention` reaches `rotary_tables`, which calls
# `get_config()` and breaks the graph. `torch.compile` on the whole model additionally hits the
# dynamic-shape tiling assert this build has. So the rule is: compile the elementwise leaves,
# leave anything touching SDPA, config, or the shape-varying RNN-T scan in eager.
#
# `dynamic=True` from the first trace, because every batch has its own (T, U): marking dynamic up
# front produces one shape-generic graph instead of recompiling per bucket. Measured over 24
# further batches of unseen shapes: 0 recompiles, 0 graph breaks, 0 new graphs.
_HOT_MODULES = (BiasNorm, TransducerJoiner, ConvModule, SwiGluFfn)

# dynamo's recompile limit is per CODE OBJECT, and one code object here serves every instance, so
# the graphs accumulate across channel widths: `BiasNorm.forward` alone needs one per distinct
# width (four -- 192/256/384/512) per (dtype, requires-grad) mode, i.e. four for the bf16 training
# path plus four more the first time _dev_metrics runs it in eval/fp32/no-grad. That is exactly
# torch's default of 8, and the next variant trips it.
#
# Exceeding the limit does not raise. Dynamo logs one warning and silently demotes that function to
# eager for the REST OF THE PROCESS, which is how a -12.6 % step turns back into a -7 % one at the
# first validation, hours into a run. These graphs are legitimately distinct -- different reduction
# widths, not guard thrash -- so the limit is raised rather than worked around. The headroom is
# deliberate: `encoder_dims` is an explicit tuning knob, and each new width costs another graph.
_RECOMPILE_LIMIT = 32


def compile_hot_modules(model: torch.nn.Module) -> int:
    """`torch.compile` the elementwise leaf modules in place. Returns how many were compiled.

    Uses `nn.Module.compile`, which installs a compiled `_call_impl` on the instance rather than
    wrapping the module in an `OptimizedModule`. That matters for checkpoints: wrapping would
    rename every key to `_orig_mod.*`, and this run has to stay loadable by the eager decode/eval
    path. Verified: `state_dict()` is unchanged (609 keys, none polluted) and a fresh uncompiled
    model round-trips it. Compilation is per instance but dynamo's code cache is keyed on the code
    object, so 147 instances share 16 graphs.

    Numerics sit at the bf16 floor. Against an fp32-eager reference the encoder gradient is 7.96 %
    off eager-bf16 and 8.40 % off compiled-bf16 -- and all of that gap is `SwiGluFfn`'s dropout,
    which inductor draws from its own philox stream: at `dropout=0.0` the compiled/eager difference
    is 3.6e-3, i.e. rounding. A different dropout realisation is a different sample, not a worse
    one. The loss itself is closer to fp32 compiled (7.2e-4) than eager (1.0e-3).
    """
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, _RECOMPILE_LIMIT
    )
    compiled = 0
    for module in model.modules():
        if isinstance(module, _HOT_MODULES):
            module.compile(dynamic=True)
            compiled += 1
    return compiled


def branch_gain_params(model: torch.nn.Module) -> list[torch.Tensor]:
    """Every `BiasNorm.log_scale` in the model, for logging `max exp(log_scale)` per window.

    This is the quantity `GradNormGuard` is structurally blind to: it reads the weight matrices,
    which stay healthy while the gain runs away. MEASURED over the 600k run's last 36k steps,
    `train/grad_norm` went 1.95 -> 201 while `train/grad_norm_guarded` went 0.82 -> 1.60, so ~99 %
    of the explosion lived in parameters the guard does not look at. Watch this scalar instead: it
    resting at `model.biasnorm_log_scale_max` is the failure, and it is visible ~25k steps before
    dev WER moves. See `BiasNorm.project` and `config/model.yaml`.

    BRANCH norms only. The trunk normalisers are `BiasNorm` too but they are a different population
    against a different bound (`model.trunk_norm_log_scale_max`), so counting them here would read
    every one of them as pinned -- their ceiling is above the branch ceiling this metric is
    measured against -- and `branch_gain_max` would report the trunk's window instead of the
    branch's. Their observable is `stack_mix/*_trunk`, which is the realized amplitude rather than
    a gain, and is strictly the better one.
    """
    return [
        m.log_scale
        for name, m in model.named_modules()
        if isinstance(m, BiasNorm) and not name.endswith("trunk_norm")
    ]


# `project()` clamps with `clamp_`, so a gain resting on the bound equals it to the last bit and an
# exact comparison would do. The tolerance is for the other way in: `forward` clamps the value it
# USES, so a checkpoint written under a wider bound stores values above the current one and still
# computes at it. Those are pinned too.
_CEILING_TOL = 1e-3


def gains_at_ceiling(gains: torch.Tensor, ceiling: float) -> torch.Tensor:
    """How many branch gains sit at (or above) `model.biasnorm_log_scale_max`.

    Returns a 0-d tensor, undetached from the device: the caller stacks it with the other logged
    scalars and syncs the whole group once, because every `.item()` drains the CUDA queue.

    A COUNT, not the max, because the max is what made this signal useless. Warm-starting from
    `bestrq_encoder.pt` inherits a population already on the bound -- 6 of 97 gains on the encoder
    this run started from -- so `max exp(log_scale)` reads at the ceiling from step 0 forever. What
    distinguishes a healthy warm start from the 2026-08-09 collapse is that the population GREW:
    one log_scale at the ceiling in step291600, three in step307800. See `GainCeilingWatch`.
    """
    return (gains >= ceiling - _CEILING_TOL).sum()


class GainCeilingWatch:
    """Reports when MORE branch gains come to rest on the bound than the run started with.

    The level test this replaces (`max_gain >= exp(bound)`) fired on step 0 of a healthy
    warm-started run and on all 2,400 log lines of every 600k-step run after it, because the
    pretrained encoder already ships gains on the bound. A signal that is always on carries no
    information, and this is the signal that led the 2026-08-09 divergence by ~25k steps -- the one
    `GradNormGuard` is structurally blind to (`train/grad_norm_guarded` moved 0.82 -> 1.60 while
    `train/grad_norm` went 1.95 -> 201).

    So the reference is the population at the first logged step, and the event is a new high-water
    mark above it. That is the shape the failure actually has: the bound does not stop the pressure,
    it relocates it, and the pressure spreading from one tensor to three is what preceded dev
    ctc-WER going 0.0818 -> 0.1136.

    Once per new level, not once per log window: the gains train ON the bound (`project` keeps them
    off `clamp`'s dead zone, so the count rattles), and a warning that repeats every 250 steps
    buries the one that mattered.

    BOTH `baseline` and `high_water` ride in the checkpoint's `extra`, for two different reasons.
    The baseline is there for the same reason `guard_norm_floor` is: a watch that re-latched its
    reference from the state it resumes INTO would adopt a diverged population as normal and never
    fire again. The high-water mark is there because the count is an INSTANTANEOUS per-window
    population and it rattles hard -- measured over the 600k run, every 20k-step window from 160k
    on runs min 0 / median 2 / max 5. Reconstructing the mark from the baseline on resume re-arms
    the watch against noise it has already reported: resume #4 at step 210,799 re-printed the
    warning at 3 and then at 5, both levels first reached ~50k steps earlier. Four resumes, four
    rounds of false alarms, on the one signal meant to lead a collapse by 25k steps.
    """

    def __init__(self, baseline: int | None = None, high_water: int | None = None) -> None:
        self._baseline = baseline
        # A pre-2026-08-26 checkpoint stores the baseline alone; falling back to it reproduces the
        # old reconstruction exactly, which is the correct reading of a run that never had a mark.
        self._high_water = baseline if high_water is None else high_water

    @property
    def baseline(self) -> int:
        """The reference population. 0 until the first `update` latches it."""
        return self._baseline or 0

    @property
    def high_water(self) -> int:
        """The worst level warned about so far. Persist it, or a resume re-warns the noise."""
        return self._high_water or 0

    def update(self, n_at_ceiling: int) -> bool:
        """Feed one logged count. Returns whether this is a new worst level worth warning about."""
        if self._baseline is None:
            self._baseline = n_at_ceiling
            if self._high_water is None:
                self._high_water = n_at_ceiling
            return False
        if n_at_ceiling <= (self._high_water or 0):
            return False
        self._high_water = n_at_ceiling
        return True


def stack_mix_params(model: torch.nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per stack, the two scalars that decide how it mixes: `(bypass, last norm_out.log_scale)`.

    A stack returns `residual + b * (x - residual)`, and `x`'s RMS is `g = exp(log_scale)` of the
    LAST block's `norm_out` -- so the output is `(1 - b) * residual + (b * g) * x_hat`. Those two
    coefficients are what the model actually chooses, and neither is visible in any scalar the
    trainer logged: `b` alone reads as a gate drifting a little, `g` alone as a norm, and only the
    pair says whether a stack is quietly turning into a processed-branch amplifier.

    MEASURED on the 600k run, `stack_mix/*` between step 298k and 310k, in the re-run that already
    had the gain capped at the old one-sided exp(2.5) = 12.18 -- the cap delayed this collapse by
    ~25k steps and did not prevent it:

        step            298k   300k   302k   304k   306k   308k   310k
        stack 1  b*g     4.52   4.83   5.07   5.98   8.80  12.18  12.18
        stack 1  1-b     0.25   0.26   0.28   0.27   0.16   0.00   0.00
        stack 2  b*g     7.00   7.19   7.18   7.04   5.53   2.58   1.18
        stack 2  1-b     0.43   0.41   0.41   0.42   0.55   0.79   0.90

    Read that as one event, not two. Stack 1 pushes its processed share to the ceiling AND drops
    its residual share to zero, so it emits RMS 12 into stack 2 -- whose blocks are all pre-normed
    and therefore contribute an O(1) correction no matter how large their input is. Stack 2 is
    silenced by arithmetic, not by choice, and its own coefficients collapse to match. dev ctc-WER
    over the same window: 0.0818 at 300k (the run's best) -> 0.1136 at 310k.

    So the cap was never the fix; it only decided which value the runaway parked at.

    NOTE WHAT THIS PAIR CANNOT SEE, because in 2026-08-22 it read healthier at the collapse than
    before it (stack 1's `b*g` 2.37 -> 0.73) while the trunk went to RMS 570. `b*g` is the
    PROCESSED share only. The residual share is `(1 - b) * in_proj(input)`, and `in_proj` has no
    normaliser on it -- so the amplitude a stack emits is not a function of these two scalars at
    all. `trunk_gain_params` is the missing half; log both.

    Ordered by `model.modules()`, i.e. stack 0..5. Returned undetached; the caller logs under
    `no_grad`.
    """
    stacks = [m for m in model.modules() if isinstance(m, ZipformerStack)]
    return [(s.bypass, cast(ZipformerBlock, s.blocks[-1]).norm_out.log_scale) for s in stacks]


def trunk_gain_max(model: torch.nn.Module) -> torch.Tensor:
    """Largest `in_proj` spectral norm over the stacks -- the trunk's worst-case gain per stack.

    The encoder's trunk is `frontend -> stack0 -> ... -> stack5 -> final_downsample -> out_norm`,
    and the only operator between two stacks is that stack's `in_proj`. Nothing normalises it, so
    until `ZipformerStack.project` existed nothing bounded it at all: MEASURED over the 2026-08-22
    run these went from init to 3x their healthy level while every other scalar the trainer logged
    -- `branch_gain_max`, `gains_at_ceiling`, `stack_mix/*`, `grad_norm_guarded` -- sat at its
    baseline. This is the one that moved.

    SPECTRAL, not `||W||_F / sqrt(n_out)`, which is what this returned first and which the run
    proved useless: that is the gain against isotropic input, the inflation runs along a few
    directions instead (sigma/isotropic 2.1-3.7 and rising), and a bound on it clipped nothing in
    43k steps while the realized trunk amplitude grew 7x. Read together with `stack_mix/*_trunk`,
    which is the realized amplitude rather than a bound on it.

    Free: the stacks each keep a warm power-iteration iterate for their own projection, so this is
    the value that projection just computed. Stacks whose `in_proj` is an `Identity`
    (`dim_in == dim`, stack 0) have unit gain by construction and are skipped. Returned as a 0-d
    device tensor so the caller can stack it with the other logged scalars and sync once.
    """
    sigmas = [
        s.trunk_sigma()
        for s in model.modules()
        if isinstance(s, ZipformerStack) and isinstance(s.in_proj, torch.nn.Linear)
    ]
    return torch.stack(sigmas).max()


def trunk_stable_rank_min(model: torch.nn.Module) -> torch.Tensor:
    """Smallest `in_proj` stable rank over the stacks -- how flat the trunk has been squeezed.

    The blind spot `trunk_gain_max` leaves and the one the 2026-08-24 collapse went through. That
    metric reads one number off the top of the spectrum, so it is constant by construction once the
    bound binds, and a projection that binds every step can hold it there while removing the rest
    of the matrix: MEASURED on `stacks.3.in_proj` with `trunk_gain_max` pinned at exactly 10.00
    from step 71,000 on, stable rank went 48.9 (75.6k) -> 32.6 (86.4k) -> 21.7 (97.2k) out of 384
    available directions, halving every ~20k steps.

    Falling here is an amplitude signal before it is a capacity signal, which is why it belongs
    next to the other two: a trunk collapsing toward rank 1 pulls its own input into the surviving
    direction, so `stack_mix/*_trunk` climbs toward `trunk_gain_max` while `trunk_gain_max` itself
    sits still. Healthy scale: 64-95 at the BEST-RQ warm start, 100-120 for the stacks of the same
    run that never bound.

    Free for the same reason `trunk_gain_max` is -- the Frobenius norm is one pass and sigma comes
    warm out of the projection's own iterate. Returned as a 0-d device tensor.
    """
    ranks = [
        s.trunk_stable_rank()
        for s in model.modules()
        if isinstance(s, ZipformerStack) and isinstance(s.in_proj, torch.nn.Linear)
    ]
    return torch.stack(ranks).min()


def trunk_rms_values(model: torch.nn.Module) -> list[torch.Tensor]:
    """Per stack, the RMS of what its `in_proj` emitted on the last forward.

    THE quantity, as opposed to a bound on it. Every constraint in `ZipformerStack.project` and
    `BiasNorm` is a proxy for this number, and no metric read it through four collapses in a row:
    at step 43,200 of the 600k run this was 5.23 / 9.97 / 43.19 / 3.06 / 3.92 against 2.68 / 2.24 /
    1.51 / 2.69 / 2.25 for the model that shipped 3.43 % test-clean, while `trunk_gain_max` read
    3.24 out of a budget of 4.0 and `stack_mix/*` read healthier than it had 20k steps earlier.

    Stack 0's `in_proj` is an `Identity`, so its entry is the frontend's output amplitude -- the
    one trunk operator no projection reaches (`frontend.linear` is unnormalised and unbounded, and
    measured 3.47 -> 9.98 across the two BEST-RQ pretrains).
    """
    stacks = [m for m in model.modules() if isinstance(m, ZipformerStack)]
    return [s.trunk_rms for s in stacks if s.trunk_rms is not None]


class GradNormGuard:
    """Detects the slow gradient-norm runaway that `clip_grad_norm_` structurally cannot catch.

    `grad_clip` is a diagnostic here, not a safety bound. Muon's Newton-Schulz iteration
    renormalises the momentum buffer to unit spectral norm, so its update magnitude is
    `lr * sqrt(max(1, m/n))` no matter how large the gradient was: scaling `p.grad` beforehand
    changes the direction not at all and the step size not at all. Every 2D hidden matrix in the
    encoder is on Muon, so clipping protects none of the model's capacity. Decoupled weight decay
    cannot bound it either: at `weight_decay 1e-2` against an encoder Muon LR of 1e-2 the
    equilibrium norm sits ~20x above where this model trains. Lowering the LR is the only live
    lever, which makes *noticing* the runaway the whole job.

    Fed from `guarded_parameters` (weight matrices only), NOT from the global norm
    `clip_grad_norm_` returns. See that function: the global norm is ~95 % the gradient of one
    scalar bypass gate, and driving the guard from it produced two false aborts. This is why the
    persisted floor key is `guard_norm_floor` and not the old `grad_norm_floor` -- the two are
    different quantities in different units, and a floor learned under the old metric must not be
    inherited by the new one.

    MEASURED on the run this exists because of (2026-08-05): `train/grad_norm` median held
    1.05-1.18 for 200k steps, then 1.30 -> 1.79 -> 11.0 over 360k -> 400k, doubling every ~2.4k
    steps, while dev ctc-WER went 0.0783 (run best) -> 0.0842 and the train loss did not visibly
    move until the norm was already past 15. That climb is now known to have been the gate, not the
    encoder; what the run's checkpoints DO show is the encoder's gradient scale collapsing 4x
    (0.72 -> 0.17) between step 400k and 415.8k as dev WER regressed. This guard is one-sided and
    would not fire on that collapse -- catching it needs a second, downward test, which no run has
    yet been instrumented long enough to size.

    Why a *floor* and not a trailing baseline: against smooth exponential growth a
    "current vs recent median" ratio is blind, because the baseline is dragged along at nearly the
    same rate and the ratio stays flat. `_floor` is a running MINIMUM of the window median, so it
    records the run's quietest regime and never rises. Growth therefore has to outrun a fixed
    reference, and the trip is inevitable rather than a race.

    Medians, not raw values, because a healthy step here is spiky: p95 ran 1.4-2.2 against a 1.1
    median and single batches touched 6. Twenty log windows of those spikes still median ~1.1.

    Deliberately NOT a per-step skip. The trip condition is a persistent trend, so the right
    response is to stop and change the LR or the weight decay, not to drop batches and continue.
    Sampling at `log_every` (where the norm is already synced for logging) keeps the detector
    off the critical path, since the whole point of holding `last_grad_norm` on-device is to avoid
    a per-step device->host sync.
    """

    def __init__(self, window: int, trend_factor: float, patience: int, floor: float = 0.0) -> None:
        self._hist: deque[float] = deque(maxlen=max(1, window))
        self._trend_factor = trend_factor
        self._patience = patience
        # Restored from the checkpoint's `extra`, so the guard is not re-armed from scratch on every
        # resume. That matters more than it looks: the run this was written for resumed 16 times,
        # and a guard relearning its floor from the current regime would adopt a diverged level as
        # "quiet" and never fire again. 0.0 means "not yet established".
        self._floor = floor
        self._trips = 0

    @property
    def floor(self) -> float:
        return self._floor

    def update(self, grad_norm: float) -> tuple[bool, float]:
        """Feed one logged grad norm. Returns (should_abort, current window median).

        The median is 0.0 until the window fills, which is also the only period the guard cannot
        fire, for `window * log_every` batches after a cold start.
        """
        self._hist.append(grad_norm)
        if len(self._hist) < (self._hist.maxlen or 1):
            return False, 0.0
        median = statistics.median(self._hist)
        # Update the floor BEFORE comparing, so the first full window is its own reference and can
        # never trip on itself.
        self._floor = median if self._floor <= 0.0 else min(self._floor, median)
        if median > self._trend_factor * self._floor:
            self._trips += 1
        else:
            self._trips = 0
        return self._trips >= self._patience, median


def _seed_all(seed: int) -> None:
    # Seed model init, worker augmentation, and batch order so the blank-collapse escape (an init-
    # sensitive knife-edge) is reproducible. torch.manual_seed also fixes the DataLoader workers'
    # per-worker seeds (PyTorch derives them from the main generator), so SpecAugment/speed-perturb
    # become deterministic too. use_deterministic_algorithms is deliberately NOT set: cuDNN's CTC
    # has no deterministic kernel and would raise.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class _Checkpointed(torch.nn.Module):
    """Wraps a stack so its forward runs under activation checkpointing."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, lengths, pad_mask, chunk_size=0):
        return torch.utils.checkpoint.checkpoint(
            self.module, x, lengths, pad_mask, chunk_size, use_reentrant=False
        )
