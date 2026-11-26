from dataclasses import dataclass


@dataclass(frozen=True)
class BuildCommonVoiceManifestCommand:
    # An extracted `cv-corpus-*/en` directory: validated.tsv, dev.tsv, test.tsv and clips/.
    corpus_dir: str
    train_out: str
    dev_out: str
    test_out: str
    # Evenly strided slice of dev, the trainer's in-loop Common Voice validation set and the
    # evaluator's tuning set. The full dev is 16k clips, which would triple validation time.
    dev_subset_out: str
    dev_subset_size: int = 2000
    num_workers: int = 16

    def __post_init__(self) -> None:
        if self.dev_subset_size <= 0:
            raise ValueError(f"dev_subset_size must be positive, got {self.dev_subset_size}")
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
