from dataclasses import dataclass, field

from src.shared_kernel.Config_Adapter import get_config


@dataclass(frozen=True)
class TransducerTrainCommand:
    # Paired 2x speed-perturbed 960 h manifest: 562,482 rows, 1,933 h. Build it with
    # scripts/build_speed_perturb_manifest.py, then extract its cache with
    # scripts/precompute_features.py. Pass --train-manifest data/manifests/train.jsonl (and
    # --train-cache-split train) for the clean 961 h corpus.
    #
    # The earlier 3-way train_sp.jsonl is NOT this manifest: it was tried as the default bundled
    # with CR-CTC and landed 1.2 WER points behind clean, and it had no cache of its own, so every
    # epoch resampled from raw audio. Here the perturbation is baked into the cache at extraction.
    train_manifest: str = "data/manifests/train_sp2.jsonl"
    dev_manifest: str = "data/manifests/dev.jsonl"
    # Split names index data/features/mel/<split>.{f16,index.npy,header.json}. Each cache carries
    # its source manifest's fingerprint and LibriSpeechDataset refuses a mismatched pair, so these
    # two must move together with the manifests above. An absent cache is a warning, not an error:
    # the dataset falls back to decoding FLAC and applying the row's `speed` itself.
    cache_dir: str = "data/features/mel"
    train_cache_split: str = "train_sp2"
    dev_cache_split: str = "dev"
    tokenizer_model: str = "data/tokenizer/bpe500.model"
    # Missing file is an ERROR here, not a silent fallback to mean 0 / std 1: that
    # fallback belongs to tests and inference. "" trains on unnormalised log-mel on purpose.
    cmvn_path: str = "data/features/cmvn.pt"
    ckpt_dir: str = "data/checkpoints"
    log_dir: str = "runs/transducer"
    total_steps: int = field(default_factory=lambda: get_config().training.transducer.total_steps)
    warm_start: str = field(default_factory=lambda: get_config().training.transducer.warm_start)
    device: str = "cuda"
    resume: bool = True
