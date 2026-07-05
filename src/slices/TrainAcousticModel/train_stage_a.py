# src/slices/TrainAcousticModel/train_stage_a.py
from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand
from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a


def main() -> None:
    ckpt = run_stage_a(StageATrainCommand())
    print(f"Stage A finished. Last checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
