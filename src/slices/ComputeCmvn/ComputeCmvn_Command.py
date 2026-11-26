from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeCmvnCommand:
    # The training sources as the trainer reads them (speed-perturbed copies and trimmed spans
    # included), pooled before sampling, so each source weighs in by its share of training frames.
    manifests: tuple[str, ...]
    cmvn_out: str
    max_utts: int | None = None
    # Random fraction of the pooled rows; global CMVN converges long before the full corpus.
    sample_frac: float | None = None
    seed: int = 0
    num_workers: int = 8

    def __post_init__(self) -> None:
        if not self.manifests:
            raise ValueError("at least one manifest is needed for CMVN statistics")
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
