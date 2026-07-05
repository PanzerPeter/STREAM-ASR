# src/slices/TrainAcousticModel/train_stage_b.py
from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand
from src.slices.TrainAcousticModel.StageBTrainer_Handler import run_stage_b


def main() -> None:
    run_stage_b(StageBTrainCommand())


if __name__ == "__main__":
    main()
