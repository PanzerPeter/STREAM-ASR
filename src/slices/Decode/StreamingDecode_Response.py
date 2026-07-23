from dataclasses import dataclass


@dataclass(frozen=True)
class NbestEntry:
    """One acoustic n-best hypothesis with its rescoring terms kept SEPARATE, so a whole
    (lm_weight, ilm_weight) grid can be ranked over a single cached decode without re-decoding.
    `lm` and `ilm` are unweighted log-probabilities; the weights are applied by the ranker."""

    ids: list[int]
    acoustic: float
    lm: float
    ilm: float


@dataclass(frozen=True)
class SegmentResult:
    text: str
    nbest: list[tuple[str, float]]


@dataclass(frozen=True)
class StreamingDecode_Response:
    text: str
    segments: list[SegmentResult]
    rtf: float
    first_partial_latency_s: float
