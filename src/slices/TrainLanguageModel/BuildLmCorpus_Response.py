from dataclasses import dataclass


@dataclass(frozen=True)
class BuildLmCorpus_Response:
    # Per source name ("librispeech", "fineweb", "cv"): lines and words written to the corpus.
    lines: dict[str, int]
    words: dict[str, int]
    heldout_dropped: int
