from dataclasses import dataclass


@dataclass(frozen=True)
class TrainLm_Command:
    train_bin: str
    val_bin: str
    out_dir: str
    max_steps: int  # cap for smoke/overfit; production uses get_config().lm.total_steps
    log_dir: str = (
        "runs/lm"  # TensorBoard scalars (train/loss, train/lr, val/ppl), mirrors Stage A/B
    )
