import json
import os
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch
import torch._dynamo
import yaml

from src.shared_kernel.Checkpoint_Adapter import load_checkpoint, save_checkpoint
from src.shared_kernel import Config_Adapter
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.slices.TrainAcousticModel.FusedBranchBlock import FusedBranchBlock
from src.slices.TrainAcousticModel.SnapshotRotation import (
    drop_snapshots_after,
    write_branch_point,
    write_rolling_snapshot,
)
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Command import (
    DevSplit,
    TransducerTrainCommand,
)
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import (
    _compile_blocks,
    _require_cmvn,
    run_transducer,
)

_TOKENIZER = "data/tokenizer/bpe500.model"


class _StubLog:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def warning(self, message: str) -> None:
        self._sink.append(message)


def _last(tmp_path) -> str:
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None)
    last = os.path.join(tmp_path, "transducer_last.pt")
    save_checkpoint(
        last, model, build_optimizer(model, get_config().optim), 42, best_wer=0.25, resume_count=3
    )
    return last


def test_rolling_snapshot_is_a_loadable_copy_of_last(tmp_path):
    last = _last(tmp_path)
    write_rolling_snapshot(str(tmp_path), last, 42, keep_last_n=2)
    snap = os.path.join(tmp_path, "transducer_step42.pt")
    assert open(snap, "rb").read() == open(last, "rb").read()
    assert not os.path.exists(snap + ".tmp")
    restored = load_checkpoint(snap, TransducerModel(cmvn_path=None))
    assert (restored["step"], restored["best_wer"], restored["resume_count"]) == (42, 0.25, 3)


def test_rolling_snapshot_prunes_but_never_touches_branch_points(tmp_path):
    # Invariant 13: the branched-cooldown trunk checkpoints are kept, never rolled off.
    last = _last(tmp_path)
    write_branch_point(str(tmp_path), last, 7)
    for step in (10, 5, 20, 15):  # out of order: pruning sorts numerically, not lexically
        write_rolling_snapshot(str(tmp_path), last, step, keep_last_n=2)
    drop_snapshots_after(str(tmp_path), 0, _StubLog([]))
    kept = sorted(f for f in os.listdir(tmp_path) if f != "transducer_last.pt")
    assert kept == ["transducer_branch7.pt"]


def test_snapshots_ahead_of_the_resume_point_are_dropped(tmp_path):
    # A snapshot left by a run that was later rolled back outranks every snapshot the replacement
    # writes, and average_checkpoints.py --last-n would average that diverged model in.
    for step in (394200, 399600, 415800):
        open(os.path.join(tmp_path, f"transducer_step{step}.pt"), "w").close()
    logged: list[str] = []
    drop_snapshots_after(str(tmp_path), 399600, _StubLog(logged))
    assert sorted(os.listdir(tmp_path)) == ["transducer_step394200.pt", "transducer_step399600.pt"]
    assert len(logged) == 1 and "step415800" in logged[0]


def test_compiled_blocks_leave_checkpoints_portable():
    # nn.Module.compile keeps state_dict keys; an OptimizedModule wrapper would rename them to
    # _orig_mod.* and break the eager Decode/Evaluate load.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None)
    before = list(model.state_dict())
    n = _compile_blocks(model)
    assert n == sum(isinstance(m, FusedBranchBlock) for m in model.modules()) == 20
    assert list(model.state_dict()) == before
    assert torch._dynamo.config.recompile_limit >= 64
    TransducerModel(cmvn_path=None).load_state_dict(model.state_dict())


def test_missing_cmvn_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cmvn"):
        _require_cmvn(str(tmp_path / "absent.pt"))


def _corpus(tmp_path, name: str, n: int) -> str:
    sr = get_config().audio.sample_rate
    rng = np.random.default_rng(0)
    texts = ["HELLO WORLD", "THE CAT SAT", "A QUICK TEST", "GOOD MORNING", "YES", "NO MORE"]
    rows = []
    for i in range(n):
        wav = (rng.standard_normal(int(sr * (1.0 + 0.3 * i))) * 0.05).astype("float32")
        path = tmp_path / f"{name}{i}.wav"
        sf.write(str(path), wav, sr)
        rows.append({"audio_filepath": str(path), "text": texts[i % 6], "num_samples": len(wav)})
    manifest = tmp_path / f"{name}.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(manifest)


@pytest.mark.skipif(not os.path.isfile(_TOKENIZER), reason="needs the BPE-500 tokenizer")
def test_run_transducer_end_to_end_on_cpu(tmp_path, monkeypatch):
    # Every phase in four steps: full-lattice warmup, the band switch, validation (label-looping
    # greedy both modes, band gap), a rolling snapshot, a branch point, then a resume.
    config = tmp_path / "config"
    shutil.copytree("config", config)
    training = yaml.safe_load((config / "training.yaml").read_text())
    training["transducer"].update(
        log_every=1,
        val_every=2,
        ckpt_every=2,
        band_warmup_steps=2,
        branch_points=[3],
        warmup_steps=1,
        compile_blocks=False,
        keep_last_n=2,
        max_frames_per_batch=1200,
    )
    (config / "training.yaml").write_text(yaml.safe_dump(training))
    monkeypatch.setattr(Config_Adapter, "_CONFIG_DIR", config)
    get_config.cache_clear()
    cmvn = tmp_path / "cmvn.pt"
    torch.save({"mean": torch.full((80,), -5.0), "std": torch.full((80,), 4.0)}, cmvn)
    cmd = TransducerTrainCommand(
        train_manifest=_corpus(tmp_path, "train", 6),
        train_cache_split="absent",
        dev_splits=(DevSplit("other", _corpus(tmp_path, "dev", 3), "absent"),),
        cache_dir=str(tmp_path / "no_cache"),
        tokenizer_model=_TOKENIZER,
        cmvn_path=str(cmvn),
        ckpt_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "runs"),
        total_steps=4,
        device="cpu",
        num_workers=0,
    )
    try:
        last = run_transducer(cmd)
        files = set(os.listdir(cmd.ckpt_dir))
        assert {"transducer_last.pt", "transducer_best.pt", "transducer_branch3.pt"} <= files
        assert "transducer_step2.pt" in files
        assert load_checkpoint(last, TransducerModel(cmvn_path=None))["step"] == 4
        from dataclasses import replace

        resumed = load_checkpoint(
            run_transducer(replace(cmd, total_steps=5)), TransducerModel(cmvn_path=None)
        )
        assert resumed["step"] == 5 and resumed["resume_count"] == 1
    finally:
        monkeypatch.undo()
        get_config.cache_clear()
