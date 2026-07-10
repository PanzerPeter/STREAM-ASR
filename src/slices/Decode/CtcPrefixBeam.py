# Standard CTC prefix beam search (Graves collapse) with optional LM shallow fusion. State
# (p_b, p_nb) per prefix persists across chunks so partial() returns a live best hypothesis
# mid-utterance. When an lm_scorer is supplied, each genuine new-token emission adds the LM's
# weighted next-token log-prob to a per-prefix cumulative bonus folded into ranking; lm_scorer=None
# leaves ranking and nbest() byte-identical to the no-LM decoder (the alpha=0 regression lock).
import math
from typing import TYPE_CHECKING, Optional, Protocol

import torch

from src.shared_kernel.Config_Adapter import get_config

if TYPE_CHECKING:
    # Type-only: the LM incremental state is an opaque list of KV caches. Referenced for typing
    # without a runtime import, so this pure search module keeps zero runtime LM-slice dependency.
    from src.slices.TrainLanguageModel.CausalGqaAttention import KvCache

_LmState = Optional[list["KvCache"]]
_NEG_INF = float("-inf")


class _StepScorer(Protocol):
    # Structural type: the beam only needs the LM's weighted next-token log-probs and the next
    # incremental state, not the concrete LmScorer class.
    def step_score(self, token: int, state: _LmState) -> tuple[torch.Tensor, list["KvCache"]]: ...


def _logsumexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CtcPrefixBeam:
    def __init__(self, blank_id: int, beam_size: int, lm_scorer: _StepScorer | None = None) -> None:
        self.blank = blank_id
        self.beam_size = beam_size
        self.lm = lm_scorer
        self.sos = get_config().model.sos_id
        self.reset()

    def reset(self) -> None:
        # prefix (tuple) -> [log p ending in blank, log p ending in non-blank]
        self.beams: dict[tuple[int, ...], list[float]] = {(): [0.0, _NEG_INF]}
        # Per-prefix cumulative LM bonus (sum of weighted next-token log-probs along the prefix) and
        # the LM decode state; lm_state[P] is the incremental state to feed P's last token from.
        self.lm_bonus: dict[tuple[int, ...], float] = {(): 0.0}
        self.lm_state: dict[tuple[int, ...], _LmState] = {(): None}

    def advance(self, log_probs: torch.Tensor) -> None:
        for t in range(log_probs.shape[0]):
            lp = log_probs[t]
            next_beams: dict[tuple[int, ...], list[float]] = {}
            next_bonus: dict[tuple[int, ...], float] = {}
            next_state: dict[tuple[int, ...], _LmState] = {}

            def _add(prefix: tuple[int, ...], pb: float, pnb: float) -> None:
                cur = next_beams.get(prefix, [_NEG_INF, _NEG_INF])
                cur[0] = _logsumexp(cur[0], pb)
                cur[1] = _logsumexp(cur[1], pnb)
                next_beams[prefix] = cur

            topk = torch.topk(lp, min(self.beam_size, lp.shape[0])).indices.tolist()
            for prefix, (p_b, p_nb) in self.beams.items():
                p_total = _logsumexp(p_b, p_nb)
                _add(prefix, p_total + float(lp[self.blank]), _NEG_INF)  # blank keeps the prefix
                if self.lm is not None and prefix not in next_bonus:
                    # A blank/merge step leaves the prefix (and its LM bonus/state) unchanged.
                    next_bonus[prefix] = self.lm_bonus[prefix]
                    next_state[prefix] = self.lm_state.get(prefix)
                # One LM step per source prefix: the next-token distribution and the resulting child
                # state are shared by every new-token expansion of this prefix.
                lm_logp: torch.Tensor | None = None
                child_state: _LmState = None
                if self.lm is not None:
                    prev = prefix[-1] if prefix else self.sos
                    lm_logp, child_state = self.lm.step_score(prev, self.lm_state.get(prefix))
                last = prefix[-1] if prefix else None
                for c in topk:
                    if c == self.blank:
                        continue
                    lpc = float(lp[c])
                    if c == last:
                        # A repeat only merges (no new token) when the prior path did NOT end in
                        # blank; a repeat after blank is a genuine second token.
                        _add(prefix, _NEG_INF, p_nb + lpc)
                        child = prefix + (c,)
                        _add(child, _NEG_INF, p_b + lpc)
                    else:
                        child = prefix + (c,)
                        _add(child, _NEG_INF, p_total + lpc)  # new distinct token
                    if lm_logp is not None and child not in next_bonus:
                        next_bonus[child] = self.lm_bonus[prefix] + float(lm_logp[c])
                        next_state[child] = child_state

            ranked = sorted(
                next_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]) + next_bonus.get(kv[0], 0.0),
                reverse=True,
            )
            self.beams = dict(ranked[: self.beam_size])
            self.lm_bonus = {p: next_bonus.get(p, 0.0) for p in self.beams}
            self.lm_state = {p: next_state.get(p) for p in self.beams}

    def nbest(self) -> list[tuple[tuple[int, ...], float]]:
        scored = [
            (p, _logsumexp(pb, pnb) + self.lm_bonus.get(p, 0.0))
            for p, (pb, pnb) in self.beams.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def partial(self) -> list[int]:
        return list(self.nbest()[0][0])
