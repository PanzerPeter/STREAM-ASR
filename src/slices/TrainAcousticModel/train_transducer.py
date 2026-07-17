# src/slices/TrainAcousticModel/train_transducer.py — CLI entry point (AC-009)
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer


def main() -> None:
    run_transducer(TransducerTrainCommand())


if __name__ == "__main__":
    main()
