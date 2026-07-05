# src/slices/BuildManifest/TrainTokenizer_Command.py — input DTO (AC-009)
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainTokenizerCommand:
    manifest: str
    model_prefix: str
    vocab_size: int
