from dataclasses import dataclass


@dataclass(frozen=True)
class TrainTokenizerCommand:
    manifest: str
    model_prefix: str
    vocab_size: int
