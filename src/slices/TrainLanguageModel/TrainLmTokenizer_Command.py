from dataclasses import dataclass


@dataclass(frozen=True)
class TrainLmTokenizer_Command:
    corpus_text: str  # the LM corpus (one normalised sentence per line), dev/test already held out
    model_prefix: str  # writes <prefix>.model and <prefix>.vocab
    vocab_size: int
    eos_id: int  # the control id reserved for end-of-line (lm.eos_id)
    # SentencePiece samples this many lines (reservoir, shuffled) instead of loading the ~1e8-line
    # corpus; a few million lines saturate an 8k unigram vocab.
    sample_lines: int = 2_000_000
