import os

import pytest
import torch

from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand
from src.slices.TrainAcousticModel.StageBTrainer_Handler import run_stage_b


@pytest.mark.slow
def test_stage_b_smoke(tmp_path):
    # 5-step run on the dev manifest, no warm-start file, exercises joint loss + chunk sampling
    # + checkpoint save end to end.
    cmd = StageBTrainCommand(
        train_manifest="data/manifests/dev.jsonl",
        dev_manifest="data/manifests/dev.jsonl",
        ckpt_dir=str(tmp_path),
        log_dir=str(tmp_path / "runs"),
        total_steps=5,
        warm_start="",  # empty -> skip warm-start (random init) for the smoke run
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    out = run_stage_b(cmd)
    assert os.path.isfile(out)
