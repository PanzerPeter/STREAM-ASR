from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class DevSplit:
    name: str  # suffix of the dev/* tags
    manifest: str
    cache_split: str


@dataclass(frozen=True)
class TrainSource:
    # A cache is indexed by its manifest's row order and carries that manifest's fingerprint, so
    # the manifest and its cache split always move together.
    manifest: str
    cache_split: str


@dataclass(frozen=True)
class TransducerTrainCommand:
    # Read side by side (one dataset per source, one sampler over their rows), so no cache is ever
    # copied or merged. LibriSpeech 960 h paired 2x speed perturbed (562,482 rows, 1,933 h) and
    # Common Voice cv-train (leakage-filtered, silence-trimmed). Sampling is per frame, so each
    # source weighs in by its hours.
    train_sources: tuple[TrainSource, ...] = (
        TrainSource("data/manifests/train_sp2.jsonl", "train_sp2"),
        TrainSource("data/manifests/cv-train.jsonl", "cv-train"),
    )
    # The last split is the fixed probe batch (band gap, gate saturation): dev-other, as before
    # Common Voice joined. cv-dev2k is an evenly strided 2,000-clip slice of cv-dev.
    dev_splits: tuple[DevSplit, ...] = (
        DevSplit("clean", "data/manifests/dev.jsonl", "dev"),
        DevSplit("cv", "data/manifests/cv-dev2k.jsonl", "cv-dev2k"),
        DevSplit("other", "data/manifests/dev-other.jsonl", "dev-other"),
    )
    cache_dir: str = "data/features/mel"
    tokenizer_model: str = "data/tokenizer/bpe500.model"
    # A missing file is an error here, not the encoder's silent mean 0 / std 1 fallback.
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/transducer"
    total_steps: int = field(default_factory=lambda: get_config().training.transducer.total_steps)
    device: str = "cuda"
    resume: bool = True
    # Train-loader worker processes; 0 loads in-process (CPU smoke tests).
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if not self.train_sources:
            raise ValueError("at least one train source is needed")
        if not self.dev_splits:
            raise ValueError("at least one dev split is needed for checkpoint selection")
