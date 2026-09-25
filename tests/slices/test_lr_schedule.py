import math

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LrSchedule import lr_at

WARMUP, TOTAL = 10_000, 175_000


def _wsd(step: int, **kw) -> float:
    return lr_at(step, 1.0, WARMUP, TOTAL, min_ratio=0.01, **kw)


def test_warmup_is_linear_to_the_peak():
    assert _wsd(0) == 0.0
    assert _wsd(WARMUP // 2) == 0.5
    assert _wsd(WARMUP) == 1.0


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


def test_raising_total_only_moves_the_decay_window():
    # The property WSD was chosen for: extending the budget mid-run must not re-heat the LR.
    step = 120_000
    assert _wsd(step) == lr_at(step, 1.0, WARMUP, 2 * TOTAL, min_ratio=0.01) == 1.0


def test_config_exposes_the_schedule_knobs():
    tr = get_config().training.transducer
    assert 0.0 < tr.lr_decay_frac < 1.0
    assert 0.0 <= tr.lr_min_ratio < 1.0
