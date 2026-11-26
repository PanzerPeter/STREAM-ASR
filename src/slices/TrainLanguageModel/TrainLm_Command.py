from dataclasses import dataclass


@dataclass(frozen=True)
class TrainLm_Command:
    train_bin: str
    val_bin: str
    out_dir: str
    # The WSD budget: the loop runs to it and the anneal ends exactly on it. Production passes
    # lm.total_steps; a cooldown branch passes its own shorter budget (COMMANDS.md, Step 5).
    total_steps: int
    log_dir: str = (
        "runs/lm"  # TensorBoard scalars (train/loss, train/lr, val/ppl), as the acoustic trainers
    )
    # Pick lm_last.pt back up if it exists, as the acoustic trainers do. `--fresh` flips it off for
    # the case that actually needs it: a config change (d_model, optimizer) whose old checkpoint
    # would either fail to load or silently continue a different recipe.
    resume: bool = True
