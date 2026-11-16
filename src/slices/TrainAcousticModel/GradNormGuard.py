import statistics
from collections import deque


class GradNormGuard:
    """Detects a slow gradient-norm runaway that clipping structurally cannot catch.

    Newton-Schulz renormalises Muon's update, so clipping the gradient bounds no step of any 2-D
    matrix; lowering the LR is the only live lever, which makes noticing the runaway the whole job.
    Fed `train/grad_norm_guarded` (weight matrices only): a per-element parameter's gradient is a
    reduction over a whole [B, T, C] activation, sized by the batch rather than by model health,
    and a guard on the global norm aborts healthy runs.

    The reference is a running MINIMUM of the window median, so it records the run's quietest
    regime and never rises: against smooth exponential growth a trailing baseline is dragged along
    and the ratio stays flat. Persisted through the checkpoint so a resume cannot re-arm against a
    diverged regime. Medians, because a healthy step is spiky.
    """

    def __init__(self, window: int, trend_factor: float, patience: int, floor: float = 0.0) -> None:
        self._hist: deque[float] = deque(maxlen=max(1, window))
        self._enabled = window > 0
        self._trend_factor = trend_factor
        self._patience = patience
        self._floor = floor  # 0.0 = not yet established
        self._trips = 0

    @property
    def floor(self) -> float:
        return self._floor

    def update(self, grad_norm: float) -> tuple[bool, float]:
        """Feed one logged norm. Returns (should_abort, window median), 0.0 until it fills."""
        if not self._enabled:
            return False, 0.0
        self._hist.append(grad_norm)
        if len(self._hist) < (self._hist.maxlen or 1):
            return False, 0.0
        median = statistics.median(self._hist)
        # Floor first, so the first full window is its own reference and cannot trip on itself.
        self._floor = median if self._floor <= 0.0 else min(self._floor, median)
        self._trips = self._trips + 1 if median > self._trend_factor * self._floor else 0
        return self._trips >= self._patience, median
