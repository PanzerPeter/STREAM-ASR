from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class DevSplit:
    name: str  # suffix of the dev/* tags
    manifest: str
    cache_split: str


@dataclass(frozen=True)
class TransducerTrainCommand:
    # Paired 2x speed-perturbed 960 h manifest (562,482 rows, 1,933 h), cache baked at extraction.
    # A cache is indexed by its manifest's row order and carries that manifest's fingerprint, so
    # the manifest and its cache split always move together.
    train_manifest: str = "data/manifests/train_sp2.jsonl"
    train_cache_split: str = "train_sp2"
    dev_splits: tuple[DevSplit, ...] = (
        DevSplit("clean", "data/manifests/dev.jsonl", "dev"),
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
        if not self.dev_splits:
            raise ValueError("at least one dev split is needed for checkpoint selection")
