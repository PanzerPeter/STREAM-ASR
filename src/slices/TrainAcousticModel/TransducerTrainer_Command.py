from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class TransducerTrainCommand:
    # Clean 960 h manifest. The 3-way speed-perturbed variant (build it with
    # scripts/build_speed_perturb_manifest.py, pass --train-manifest data/manifests/train_sp.jsonl)
    # was tried as the default and is not it: the run that used it, together with CR-CTC, landed
    # 1.2 WER points behind this manifest. It also forces the mel cache off, since a perturbed row
    # has to be resampled from raw audio.
    train_manifest: str = "data/manifests/train.jsonl"
    dev_manifest: str = "data/manifests/dev.jsonl"
    tokenizer_model: str = "data/tokenizer/bpe500.model"
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/transducer"
    total_steps: int = field(default_factory=lambda: get_config().training.transducer.total_steps)
    warm_start: str = field(default_factory=lambda: get_config().training.transducer.warm_start)
    device: str = "cuda"
    resume: bool = True
