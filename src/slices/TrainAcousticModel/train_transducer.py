# Transducer training entry point (GPU; user-run).
import argparse
from dataclasses import replace

from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer


def _parse_args() -> TransducerTrainCommand:
    # Defaults come straight from the Command DTO (which itself reads config), so a bare invocation
    # is the production recipe and every knob has a flag -- no dataclasses.replace one-liners for a
    # fresh run or a smoke run.
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Train the single-pass streaming RNN-T transducer.")
    p.add_argument("--train-manifest", default=base.train_manifest)
    p.add_argument("--dev-manifest", default=base.dev_manifest)
    # A cache is indexed by its manifest's row order and carries that manifest's fingerprint, so
    # these travel with the two flags above. Mismatched pairs raise at construction rather than
    # training on the wrong audio; '' opts out of the cache and decodes FLAC per epoch.
    p.add_argument("--train-cache-split", default=base.train_cache_split)
    p.add_argument("--dev-cache-split", default=base.dev_cache_split)
    p.add_argument("--cache-dir", default=base.cache_dir)
    p.add_argument("--warm-start", default=base.warm_start, help="encoder ckpt; '' = from scratch")
    p.add_argument("--total-steps", type=int, default=base.total_steps)
    p.add_argument("--log-dir", default=base.log_dir)
    p.add_argument("--ckpt-dir", default=base.ckpt_dir)
    p.add_argument("--device", default=base.device)
    p.add_argument(
        "--fresh", action="store_true", help="ignore any *_last.pt and start from step 0"
    )
    a = p.parse_args()
    return replace(
        base,
        train_manifest=a.train_manifest,
        dev_manifest=a.dev_manifest,
        train_cache_split=a.train_cache_split,
        dev_cache_split=a.dev_cache_split,
        cache_dir=a.cache_dir,
        warm_start=a.warm_start,
        total_steps=a.total_steps,
        log_dir=a.log_dir,
        ckpt_dir=a.ckpt_dir,
        device=a.device,
        resume=not a.fresh,
    )


def main() -> None:
    run_transducer(_parse_args())


if __name__ == "__main__":
    main()
