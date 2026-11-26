from dataclasses import dataclass


@dataclass(frozen=True)
class TrainTokenizerCommand:
    # Every training source's transcripts, so no source's vocabulary is spelled out letter by
    # letter. The base train manifest, never its speed-perturbed copy: that would count each
    # transcript twice.
    manifests: tuple[str, ...]
    model_prefix: str
    vocab_size: int

    def __post_init__(self) -> None:
        if not self.manifests:
            raise ValueError("at least one manifest is needed to train a tokenizer")
