import math

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LrSchedule import lr_at

WARMUP, TOTAL = 10_000, 175_000


def _wsd(step: int, **kw) -> float:
    return lr_at(step, 1.0, WARMUP, TOTAL, schedule="wsd", min_ratio=0.01, **kw)


def test_warmup_is_linear_to_the_stable_level():
    assert _wsd(0) == 0.0
    assert _wsd(WARMUP // 2) == 0.5
    assert _wsd(WARMUP) == 1.0
    assert _wsd(WARMUP // 2, stable_ratio=0.5) == 0.25


def test_wsd_holds_then_anneals_to_the_floor():
    decay_start = TOTAL - round(0.25 * TOTAL)
    assert _wsd(WARMUP + 1) == 1.0
    assert _wsd(decay_start - 1) == 1.0
    # The 1-sqrt profile is front-loaded: half the anneal window is already most of the way down.
    assert _wsd(decay_start) == 1.0
    assert _wsd(decay_start + (TOTAL - decay_start) // 2) < 0.31
    assert math.isclose(_wsd(TOTAL), 0.01)
    assert math.isclose(_wsd(TOTAL + 5_000), 0.01)  # clamped, never negative


def test_wsd_is_monotone_non_increasing_after_warmup():
    values = [_wsd(s) for s in range(WARMUP, TOTAL + 1, 500)]
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_stable_ratio_pins_the_hold_level_for_a_mid_run_switch():
    # Switching an in-flight cosine run to WSD must not re-heat: at step 92k the cosine shape is
    # ~0.5, so stable_ratio 0.5 continues at exactly the LR that run already had.
    cosine_at_92k = 0.5 * (1 + math.cos(math.pi * (92_000 - WARMUP) / (TOTAL - WARMUP)))
    assert math.isclose(_wsd(92_000, stable_ratio=cosine_at_92k), cosine_at_92k)


def test_cosine_floor_replaces_the_decay_to_exactly_zero():
    plain = lr_at(TOTAL, 1.0, WARMUP, TOTAL)
    floored = lr_at(TOTAL, 1.0, WARMUP, TOTAL, min_ratio=0.01)
    assert plain == 0.0
    assert math.isclose(floored, 0.01)
    # min_ratio 0 keeps the historical curve bit-for-bit, so old runs stay reproducible.
    mid = 0.5 * (1 + math.cos(math.pi * (80_000 - WARMUP) / (TOTAL - WARMUP)))
    assert math.isclose(lr_at(80_000, 1.0, WARMUP, TOTAL), mid)


def test_config_exposes_the_schedule_knobs():
    tr = get_config().training.transducer
    assert tr.lr_schedule in ("cosine", "wsd")
    assert 0.0 < tr.lr_stable_ratio <= 1.0
    assert 0.0 < tr.lr_decay_frac < 1.0
    assert 0.0 <= tr.lr_min_ratio < 1.0
