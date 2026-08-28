import math

import torch

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.GradientClipping import (
    clip_grads_per_tensor,
    grad_norm_of,
    guarded_parameters,
    unguarded_parameters,
)
from src.slices.TrainAcousticModel._train_utils import GradNormGuard


def _guard(window: int = 20, factor: float = 4.0, patience: int = 3, floor: float = 0.0):
    return GradNormGuard(window, factor, patience, floor=floor)


def test_silent_until_the_window_fills():
    # A cold start must not fire on partial evidence: the first `window` samples establish
    # the floor and nothing else. An outright runaway during that period still returns no median.
    guard = _guard(window=20)
    for i in range(19):
        abort, median = guard.update(2.0**i)
        assert not abort and median == 0.0


def test_flat_run_never_trips():
    # The healthy regime this was sized against: median ~1.1, p95 ~2.2, single batches at 6.
    # 40 windows of that spikiness must not produce a single trip.
    guard = _guard()
    spiky = [1.05, 1.18, 0.95, 2.20, 1.10, 1.02, 6.26, 1.15, 1.08, 1.90]
    for i in range(400):
        abort, _ = guard.update(spiky[i % len(spiky)])
        assert not abort


def test_trips_on_the_measured_runaway():
    # Replays the real 2026-08-05 trajectory (train/grad_norm medians per 20k-step bin):
    # 200k steps flat, then the climb. The guard must fire during the climb and, crucially, well
    # before the norm reaches the 131-249 range where the model was already destroyed.
    guard = _guard()
    for _ in range(200):  # the long flat regime, 140k-340k
        assert not guard.update(1.12)[0]
    tripped_at = None
    for value in (1.30, 1.79, 3.19, 5.84, 15.2, 18.6, 131.7, 249.4):
        for _ in range(20):  # one full window at each level
            abort, _ = guard.update(value)
            if abort and tripped_at is None:
                tripped_at = value
    assert tripped_at is not None, "guard missed the runaway it exists for"
    assert tripped_at <= 5.84, f"fired only at {tripped_at}, far too late to save the model"


def test_smooth_exponential_growth_is_caught():
    # The failure mode a trailing baseline has: against smooth exponential growth, "current vs
    # recent median" stays flat: the baseline is dragged along at the same rate. A running-minimum
    # floor cannot be dragged, so growth has to outrun a fixed reference. Doubling every 20 windows,
    # matching the measured ~2.4k-step doubling time at log_every 250.
    guard = _guard()
    for _ in range(20):
        guard.update(1.0)
    fired = False
    for i in range(200):
        abort, _ = guard.update(2.0 ** (i / 20.0))
        fired = fired or abort
    assert fired, "a trailing-baseline guard would miss this; the floor exists to catch it"


def test_floor_survives_a_resume_into_a_degraded_regime():
    # The run this guards resumed 16 times. A guard that relearns its floor from the regime it
    # resumes INTO would adopt a diverged level as "quiet" and never fire again -- so the floor
    # is carried in the checkpoint's `extra`. Same stream, with and without the inherited floor.
    fresh = _guard()
    inherited = _guard(floor=1.12)
    for _ in range(60):
        fresh.update(6.0)
        inherited.update(6.0)
    assert not fresh.update(6.0)[0], "a cold guard cannot know 6.0 is abnormal"
    assert inherited.update(6.0)[0], "an inherited floor must still recognise the runaway"


def test_floor_is_a_running_minimum_and_never_rises():
    guard = _guard(window=4)
    for value in (5.0, 5.0, 5.0, 5.0):
        guard.update(value)
    assert math.isclose(guard.floor, 5.0)
    for value in (1.0, 1.0, 1.0, 1.0):
        guard.update(value)
    assert math.isclose(guard.floor, 1.0)
    for value in (3.0, 3.0, 3.0, 3.0):
        guard.update(value)
    assert math.isclose(guard.floor, 1.0), "floor must not follow the run back up"


def test_patience_requires_consecutive_windows():
    # One anomalous window is a bad bucket, not a trend: the counter must reset on any quiet window.
    guard = _guard(window=1, patience=3)
    guard.update(1.0)
    assert not guard.update(10.0)[0]
    assert not guard.update(10.0)[0]
    assert not guard.update(1.0)[0]  # resets
    assert not guard.update(10.0)[0]
    assert not guard.update(10.0)[0]
    assert guard.update(10.0)[0]


def test_guarded_parameters_excludes_the_size_dominated_scalars():
    # The whole reason the guard aborted two healthy runs: a scalar gate multiplying a [B,T,C]
    # activation carries a gradient summed over millions of terms, so it dominates the global norm
    # by a factor set by the batch shape rather than by anything about the model.
    model = torch.nn.Module()
    model.gate = torch.nn.Parameter(torch.tensor(0.5))  # a ZipformerStack.bypass
    model.bias = torch.nn.Parameter(torch.zeros(8))  # a bias / BiasNorm.log_scale
    model.weight = torch.nn.Parameter(torch.zeros(8, 8))  # a real weight matrix
    names = {id(p): n for n, p in model.named_parameters()}
    assert [names[id(p)] for p in guarded_parameters(model)] == ["weight"]

    model.gate.grad = torch.tensor(300.0)
    model.bias.grad = torch.full((8,), 3.0)
    model.weight.grad = torch.full((8, 8), 0.25)  # norm 2.0
    guarded = float(grad_norm_of(guarded_parameters(model)))
    everything = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))
    assert math.isclose(guarded, 2.0, rel_tol=1e-6)
    assert everything > 100 * guarded, "the global norm is the scalar, as measured on the real run"


def test_the_two_parameter_sets_partition_the_model():
    model = torch.nn.Module()
    model.gate = torch.nn.Parameter(torch.tensor(0.5))
    model.bias = torch.nn.Parameter(torch.zeros(8))
    model.weight = torch.nn.Parameter(torch.zeros(8, 8))
    names = {id(p): n for n, p in model.named_parameters()}
    assert [names[id(p)] for p in unguarded_parameters(model)] == ["gate", "bias"]
    both = {id(p) for p in guarded_parameters(model)} | {id(p) for p in unguarded_parameters(model)}
    assert both == {id(p) for p in model.parameters()}, "a clipped-twice or unclipped parameter"


def test_split_clip_keeps_the_matrices_off_the_gate_gradient():
    # The amplifier bug. One clip over model.parameters() rescales every gradient by
    # grad_clip / global_norm, and the global norm is the gate -- so a gate gradient that grows
    # exponentially silently divides every weight matrix's gradient by a step-varying factor.
    # Muon's Newton-Schulz is scale-invariant WITHIN a step but its momentum buffer is an EMA
    # ACROSS steps, so that factor reweights it. MEASURED: Muon momentum norms fell 1.30 -> 0.19
    # while grad_norm_guarded held ~0.9.
    def fresh():
        model = torch.nn.Module()
        model.gate = torch.nn.Parameter(torch.tensor(0.5))
        model.weight = torch.nn.Parameter(torch.zeros(8, 8))
        model.gate.grad = torch.tensor(500.0)
        model.weight.grad = torch.full((8, 8), 0.25)  # norm 2.0, far under grad_clip
        return model

    single = fresh()
    torch.nn.utils.clip_grad_norm_(single.parameters(), 5.0)
    assert float(grad_norm_of([single.weight])) < 0.05, "this is the bug: 0.9 -> 0.009 of itself"

    split = fresh()
    torch.nn.utils.clip_grad_norm_(unguarded_parameters(split), 5.0)
    torch.nn.utils.clip_grad_norm_(guarded_parameters(split), 5.0)
    assert math.isclose(float(grad_norm_of([split.weight])), 2.0, rel_tol=1e-6)
    assert math.isclose(float(split.gate.grad), 5.0, rel_tol=1e-6), "the gate is still bounded"


def test_per_tensor_clip_does_not_let_one_scalar_rescale_the_others():
    # Splitting the clip in two was not enough: inside the scalar group, ONE gate still owned the
    # norm and therefore set the factor every other scalar's gradient was multiplied by. MEASURED
    # off transducer_last.pt's AdamW moments, encoder.stacks.3.bypass sat at exactly grad_clip
    # (4.9998 against a bound of 5.0) while the other five gates were at 0.007-0.025 -- i.e. the
    # group clip was firing on that one tensor every single step.
    def fresh():
        gate = torch.nn.Parameter(torch.tensor(0.5))
        gain = torch.nn.Parameter(torch.tensor(0.0))
        bias = torch.nn.Parameter(torch.zeros(8))
        gate.grad = torch.tensor(500.0)
        gain.grad = torch.tensor(0.02)
        bias.grad = torch.full((8,), 0.25)  # norm sqrt(8) * 0.25 = 0.707
        return [gate, gain, bias]

    shared = fresh()
    torch.nn.utils.clip_grad_norm_(shared, 5.0)
    assert float(shared[1].grad) < 1e-3, "this is the bug: 0.02 -> 0.0002 because of another tensor"

    per_tensor = fresh()
    pre = clip_grads_per_tensor(per_tensor, 5.0)
    assert math.isclose(float(per_tensor[0].grad), 5.0, rel_tol=1e-5), "the loud one is bounded"
    assert math.isclose(float(per_tensor[1].grad), 0.02, rel_tol=1e-5), "the quiet one is untouched"
    assert math.isclose(float(per_tensor[2].grad.norm()), 0.7071, rel_tol=1e-4)
    # The reported norm is the group's PRE-clip global norm, so train/grad_norm stays comparable
    # with every run logged before this change.
    assert math.isclose(float(pre), math.sqrt(500.0**2 + 0.02**2 + 0.5), rel_tol=1e-6)


def test_per_tensor_clip_survives_missing_and_zero_gradients():
    # The OOM path drops a partial accumulation window, so p.grad is None; and a gate can land on
    # exactly zero gradient. Neither may produce a nan through the 1/norm.
    live = torch.nn.Parameter(torch.zeros(4))
    zero = torch.nn.Parameter(torch.zeros(4))
    absent = torch.nn.Parameter(torch.zeros(4))
    live.grad = torch.full((4,), 10.0)
    zero.grad = torch.zeros(4)
    clip_grads_per_tensor([live, zero, absent], 1.0)
    assert math.isclose(float(live.grad.norm()), 1.0, rel_tol=1e-5)
    assert torch.equal(zero.grad, torch.zeros(4))
    assert absent.grad is None
    assert float(clip_grads_per_tensor([absent], 1.0)) == 0.0


def test_grad_norm_of_ignores_parameters_without_gradients():
    # The OOM path drops a partial accumulation window, and a warm-started module can sit unused
    # for a whole step; either leaves p.grad as None. Those must not crash the reduction.
    a = torch.nn.Parameter(torch.zeros(4, 4))
    b = torch.nn.Parameter(torch.zeros(4, 4))
    a.grad = torch.full((4, 4), 0.5)  # norm 2.0
    assert math.isclose(float(grad_norm_of([a, b])), 2.0, rel_tol=1e-6)
    assert float(grad_norm_of([b])) == 0.0


def test_config_exposes_the_guard_knobs():
    tr = get_config().training.transducer
    assert tr.guard_window == 20
    assert tr.guard_trend_factor == 4.0
    assert tr.guard_patience == 3
