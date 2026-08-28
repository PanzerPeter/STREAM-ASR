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
    # The two terms `rtf` is the ratio of, kept separately: a corpus RTF must be
    # sum(decode_s)/sum(audio_s), because a mean of per-utterance ratios weights a 2 s utterance
    # the same as a 30 s one and LibriSpeech durations span more than an order of magnitude.
    decode_s: float
    audio_s: float
    # Search + rescoring, i.e. everything after the encoder consumed the last chunk: what a live
    # session still owes the user once the audio stops. LM n-best rescoring lands here, so this is
    # where the rescorer's latency cost shows up as itself.
    finalize_s: float
    # Wall time to the first encoder output of a streaming session. None offline: an offline pass
    # emits no partials, and reporting 0.0 makes "not measurable" look like "instant".
    first_partial_latency_s: float | None
