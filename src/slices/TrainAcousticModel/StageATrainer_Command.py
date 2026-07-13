# src/slices/TrainAcousticModel/StageATrainer_Command.py — input DTO (AC-009)
from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class StageATrainCommand:
    train_manifest: str = "data/manifests/train.jsonl"
    dev_manifest: str = "data/manifests/dev.jsonl"
    tokenizer_model: str = "data/tokenizer/bpe500.model"
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/stage_a"
    total_steps: int = field(default_factory=lambda: get_config().training.stage_a.total_steps)
    device: str = "cuda"
    # Eager is the validated default: torch.compile currently hits inductor bugs on this
    # torch 2.11 + Blackwell build (partitioner crash with checkpointing, tiling assertion
    # with dynamic shapes). Opt in to compile once a torch update fixes those.
    compile_model: bool = False
    resume: bool = True  # continue from ckpt_dir/stage_a_last.pt if present; False forces fresh
    encoder_init: str | None = None  # BEST-RQ bestrq_encoder.pt to warm-start the encoder from
