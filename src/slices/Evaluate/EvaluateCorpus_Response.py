from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateCorpus_Response:
    stage: str
    mode: str
    wer: float
    cer: float
    num_utts: int
    # The raw counts behind wer/cer. Published so two runs can be compared for significance without
    # re-decoding -- a 0.1-point WER move over 2,620 test-clean utterances is ~50 words.
    num_word_errors: int
    num_ref_words: int
    num_char_errors: int
    num_ref_chars: int
    audio_s: float
    # None unless this pass had the GPU to itself (EvaluateCorpus_Command.measure_timing).
    # rtf = sum(decode_s)/sum(audio_s); latency_s is streaming-only (offline emits no partials);
    # finalize_s is the mean post-encoder search+rescore cost per utterance.
    rtf: float | None
    latency_s: float | None
    finalize_s: float | None
