from dataclasses import dataclass


@dataclass(frozen=True)
class TrainLm_Command:
    train_bin: str
    val_bin: str
    out_dir: str
    max_steps: int  # cap for smoke/overfit; production uses get_config().lm.total_steps
    log_dir: str = (
        "runs/lm"  # TensorBoard scalars (train/loss, train/lr, val/ppl), as the acoustic trainers
    )
    # Pick lm_last.pt back up if it exists, as the acoustic trainers do. `--fresh` flips it off for
    # the case that actually needs it: a config change (d_model, optimizer) whose old checkpoint
    # would either fail to load or silently continue a different recipe.
    resume: bool = True
