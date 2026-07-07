from dataclasses import dataclass


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
