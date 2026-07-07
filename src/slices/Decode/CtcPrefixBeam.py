# Standard CTC prefix beam search (Graves collapse), no LM. State (p_b, p_nb) per prefix
# persists across chunks so partial() returns a live best hypothesis mid-utterance.
import math

import torch

_NEG_INF = float("-inf")


def _logsumexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CtcPrefixBeam:
    def __init__(self, blank_id: int, beam_size: int) -> None:
        self.blank = blank_id
        self.beam_size = beam_size
        self.reset()

    def reset(self) -> None:
        # prefix (tuple) -> [log p ending in blank, log p ending in non-blank]
        self.beams: dict[tuple[int, ...], list[float]] = {(): [0.0, _NEG_INF]}

    def advance(self, log_probs: torch.Tensor) -> None:
        for t in range(log_probs.shape[0]):
            lp = log_probs[t]
            next_beams: dict[tuple[int, ...], list[float]] = {}

            def _add(prefix: tuple[int, ...], pb: float, pnb: float) -> None:
                cur = next_beams.get(prefix, [_NEG_INF, _NEG_INF])
                cur[0] = _logsumexp(cur[0], pb)
                cur[1] = _logsumexp(cur[1], pnb)
                next_beams[prefix] = cur

            topk = torch.topk(lp, min(self.beam_size, lp.shape[0])).indices.tolist()
            for prefix, (p_b, p_nb) in self.beams.items():
                p_total = _logsumexp(p_b, p_nb)
                _add(prefix, p_total + float(lp[self.blank]), _NEG_INF)  # blank keeps the prefix
                last = prefix[-1] if prefix else None
                for c in topk:
                    if c == self.blank:
                        continue
                    lpc = float(lp[c])
                    if c == last:
                        # A repeat only merges (no new token) when the prior path did NOT end in
                        # blank; a repeat after blank is a genuine second token.
                        _add(prefix, _NEG_INF, p_nb + lpc)
                        _add(prefix + (c,), _NEG_INF, p_b + lpc)
                    else:
                        _add(prefix + (c,), _NEG_INF, p_total + lpc)  # new distinct token

            ranked = sorted(
                next_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )
            self.beams = dict(ranked[: self.beam_size])

    def nbest(self) -> list[tuple[tuple[int, ...], float]]:
        scored = [(p, _logsumexp(pb, pnb)) for p, (pb, pnb) in self.beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def partial(self) -> list[int]:
        return list(self.nbest()[0][0])
