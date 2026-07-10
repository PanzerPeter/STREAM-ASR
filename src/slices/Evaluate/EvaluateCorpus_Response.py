from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateCorpus_Response:
    stage: str
    mode: str
    wer: float
    cer: float
    rtf: float
    latency_s: float
    num_utts: int
