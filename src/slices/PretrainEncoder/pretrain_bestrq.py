from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.BestRqPretrainer_Handler import run_pretrain


def main() -> None:
    out = run_pretrain(BestRqPretrainCommand())
    print(f"BEST-RQ pretrain finished. Encoder checkpoint: {out}")


if __name__ == "__main__":
    main()
