# STREAM-LM training entry point (GPU; user-run).
import argparse

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command
from src.slices.TrainLanguageModel.TrainLm_Handler import TrainLm_Handler


def main() -> None:
    lm = get_config().lm
    p = argparse.ArgumentParser(description="Train STREAM-LM (rescoring language model).")
    p.add_argument("--train-bin", default="data/lm_data/train.bin")
    p.add_argument("--val-bin", default="data/lm_data/val.bin")
    p.add_argument("--out-dir", default="data/checkpoints")
    p.add_argument(
        "--max-steps",
        type=int,
        default=lm.total_steps,
        help="step cap; defaults to config lm.total_steps. Lower it for a smoke run.",
    )
    p.add_argument(
        "--fresh", action="store_true", help="ignore any lm_last.pt and start from step 0"
    )
    a = p.parse_args()
    cmd = TrainLm_Command(
        train_bin=a.train_bin,
        val_bin=a.val_bin,
        out_dir=a.out_dir,
        max_steps=a.max_steps,
        resume=not a.fresh,
    )
    best = TrainLm_Handler().run(cmd)
    print(f"STREAM-LM done. best val perplexity = {best:.3f} -> {a.out_dir}/lm_best.pt")


if __name__ == "__main__":
    main()
