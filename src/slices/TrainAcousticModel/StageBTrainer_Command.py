# src/slices/TrainAcousticModel/StageBTrainer_Command.py — input DTO (AC-009)
from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class StageBTrainCommand:
    train_manifest: str = "data/manifests/train.jsonl"
    dev_manifest: str = "data/manifests/dev.jsonl"
    tokenizer_model: str = "data/tokenizer/bpe500.model"
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/stage_b"
    total_steps: int = field(default_factory=lambda: get_config().training.stage_b.total_steps)
    warm_start: str = field(default_factory=lambda: get_config().training.stage_b.warm_start)
    device: str = "cuda"
    resume: bool = True
