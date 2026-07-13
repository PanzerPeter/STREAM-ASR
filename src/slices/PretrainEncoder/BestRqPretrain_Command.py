# src/slices/PretrainEncoder/BestRqPretrain_Command.py — input DTO (AC-009)
from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class BestRqPretrainCommand:
    train_manifest: str = "data/manifests/train.jsonl"
    cache_dir: str = "data/features/mel"
    cache_split: str = "train"
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/bestrq"
    total_steps: int = field(default_factory=lambda: get_config().pretrain.total_steps)
    device: str = "cuda"
    resume: bool = True
    # DataLoader worker processes; 0 forces single-process loading (CPU smoke test: forking after
    # torch/OpenMP threads are live deadlocks — same footgun SP1's precompute_features hit).
    num_workers: int = 2
    # Test hook: stop after N optimizer steps so the smoke test exercises the full loop cheaply.
    max_steps_smoke: int | None = None
