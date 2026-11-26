# Transducer training entry point (GPU; user-run).
import argparse
from dataclasses import replace

from src.slices.TrainAcousticModel.TransducerTrainer_Command import (
    TrainSource,
    TransducerTrainCommand,
)
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer


def _parse_args() -> TransducerTrainCommand:
    # Defaults come from the Command DTO (which reads config), so a bare invocation is the
    # release recipe.
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Train the S2 streaming RNN-T transducer.")
    # Repeatable and paired in order: the n-th --train-manifest is read with the n-th
    # --train-cache-split. Giving them replaces the default source list.
    p.add_argument("--train-manifest", action="append", default=[])
    p.add_argument("--train-cache-split", action="append", default=[])
    p.add_argument("--cache-dir", default=base.cache_dir)
    p.add_argument("--total-steps", type=int, default=base.total_steps)
    p.add_argument("--log-dir", default=base.log_dir)
    p.add_argument("--ckpt-dir", default=base.ckpt_dir)
    p.add_argument("--device", default=base.device)
    p.add_argument("--fresh", action="store_true", help="ignore transducer_last.pt, start at 0")
    a = p.parse_args()
    if len(a.train_manifest) != len(a.train_cache_split):
        p.error("--train-manifest and --train-cache-split must be given the same number of times")
    sources = tuple(TrainSource(m, c) for m, c in zip(a.train_manifest, a.train_cache_split))
    return replace(
        base,
        train_sources=sources or base.train_sources,
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
