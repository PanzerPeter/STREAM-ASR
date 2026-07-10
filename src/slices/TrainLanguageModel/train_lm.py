# src/slices/TrainLanguageModel/train_lm.py — STREAM-LM training entry point (GPU; user-run).
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command
from src.slices.TrainLanguageModel.TrainLm_Handler import TrainLm_Handler


def main() -> None:
    lm = get_config().lm
    cmd = TrainLm_Command(
        train_bin="data/lm_data/train.bin",
        val_bin="data/lm_data/val.bin",
        out_dir="data/checkpoints",
        max_steps=lm.total_steps,
    )
    best = TrainLm_Handler().run(cmd)
    print(f"STREAM-LM done. best val perplexity = {best:.3f} -> data/checkpoints/lm_best.pt")


if __name__ == "__main__":
    main()
