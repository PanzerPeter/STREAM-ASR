from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class BestRqPretrainCommand:
    # Paired 2x speed-perturbed 960 h manifest (562,482 rows, 1,933 h), the same corpus the
    # transducer stage trains on. At the configured budget that is 5.2 passes over 1,933 h rather
    # than 10.4 over 961 h -- for a self-supervised objective whose targets are computed from the
    # perturbed mel itself, more distinct audio beats more repeats of the same audio.
    train_manifest: str = "data/manifests/train_sp2.jsonl"
    cache_dir: str = "data/features/mel"
    cache_split: str = "train_sp2"
    # Held-out split for the periodic masked-prediction probe. Nothing else in this stage measures
    # generalisation: the train loss is over targets drawn from the very utterances just seen.
    dev_manifest: str = "data/manifests/dev.jsonl"
    dev_cache_split: str = "dev"
    # Missing file is an ERROR here, not a silent fallback to mean 0 / std 1: that
    # fallback belongs to tests and inference. "" pretrains on unnormalised log-mel on purpose.
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/bestrq"
    total_steps: int = field(default_factory=lambda: get_config().pretrain.total_steps)
    device: str = "cuda"
    resume: bool = True
    # DataLoader worker processes; 0 forces single-process loading (CPU smoke test: forking after
    # torch/OpenMP threads are live deadlocks, the same footgun precompute_features hit).
    num_workers: int = 2
    # Test hook: stop after N optimizer steps so the smoke test exercises the full loop cheaply.
    max_steps_smoke: int | None = None
