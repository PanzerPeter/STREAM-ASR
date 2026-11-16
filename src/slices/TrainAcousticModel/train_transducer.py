# Transducer training entry point (GPU; user-run).
import argparse
from dataclasses import replace

from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer


def _parse_args() -> TransducerTrainCommand:
    # Defaults come from the Command DTO (which reads config), so a bare invocation is the
    # release recipe.
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Train the S2 streaming RNN-T transducer.")
    p.add_argument("--train-manifest", default=base.train_manifest)
    p.add_argument("--train-cache-split", default=base.train_cache_split)
    p.add_argument("--cache-dir", default=base.cache_dir)
    p.add_argument("--total-steps", type=int, default=base.total_steps)
    p.add_argument("--log-dir", default=base.log_dir)
    p.add_argument("--ckpt-dir", default=base.ckpt_dir)
    p.add_argument("--device", default=base.device)
    p.add_argument("--fresh", action="store_true", help="ignore transducer_last.pt, start at 0")
    a = p.parse_args()
    return replace(
        base,
        train_manifest=a.train_manifest,
        train_cache_split=a.train_cache_split,
        cache_dir=a.cache_dir,
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
