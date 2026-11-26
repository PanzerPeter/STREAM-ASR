import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from src.slices.TrainLanguageModel.LmDataset import LmDataset

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command
from src.slices.TrainLanguageModel.TrainLm_Handler import TrainLm_Handler

_TINY_LM = {
    "d_model": 64,
    "layers": 2,
    "heads": 4,
    "kv_groups": 2,
    "ffn_expansion": 2,
    "context_len": 32,
    "batch_size": 16,
    "warmup_steps": 5,
    "eval_interval": 100,
    "lr_peak": 1.0e-3,
}


def test_loader_reads_every_window_once_per_pass(tmp_path):
    # A one-pass budget must read every token once: non-overlapping windows, shuffled without
    # replacement. (Random offsets with replacement leave e^-1 of the tokens unread.)
    bin_path = tmp_path / "toy.bin"
    bin_path.write_bytes(np.arange(2, 502, dtype=np.uint16).tobytes())
    loader = TrainLm_Handler()._loader(str(bin_path), ctx=8, batch=4, seed=0)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.replacement is False
    seen = sorted(int(x[0]) for xb, _, _ in loader for x in xb)
    n = len(LmDataset(str(bin_path), 8))
    assert len(seen) == n // 4 * 4 and len(set(seen)) == len(seen)  # drop_last trims < one batch
    assert all((s - 2) % 8 == 0 for s in seen)


def test_loss_is_cross_entropy_plus_z_loss_on_one_normaliser():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11, dtype=torch.bfloat16)
    y = torch.randint(0, 11, (2, 5))
    ce, loss = TrainLm_Handler._loss(logits, y, 1e-2)
    flat = logits.float().reshape(-1, 11)
    torch.testing.assert_close(ce, torch.nn.functional.cross_entropy(flat, y.reshape(-1)))
    torch.testing.assert_close(loss, ce + 1e-2 * flat.logsumexp(-1).pow(2).mean())
    ce0, loss0 = TrainLm_Handler._loss(logits, y, 0.0)
    assert torch.equal(ce0, loss0)


@pytest.mark.slow
def test_lm_overfits_tiny_corpus(tmp_path, monkeypatch):
    # Shrink the model + schedule to a CPU-seconds smoke that still genuinely trains: a tiny
    # deep-narrow LM must memorize a period-10 token pattern, so val perplexity collapses toward 1.
    # Production config is untouched; monkeypatch reverts these fields after the test.
    lm = get_config().lm
    for field, value in _TINY_LM.items():
        monkeypatch.setattr(lm, field, value)

    # Both bins must exceed context_len so LmDataset yields at least one window.
    train = np.tile(np.arange(10, dtype=np.uint16), 200)  # 2000 tokens
    val = np.tile(np.arange(10, dtype=np.uint16), 20)  # 200 tokens > context_len
    (tmp_path / "train.bin").write_bytes(train.tobytes())
    (tmp_path / "val.bin").write_bytes(val.tobytes())
    cmd = TrainLm_Command(
        train_bin=str(tmp_path / "train.bin"),
        val_bin=str(tmp_path / "val.bin"),
        out_dir=str(tmp_path / "ckpt"),
        total_steps=300,
        log_dir=str(tmp_path / "runs"),  # keep TensorBoard events out of the repo's runs/
    )
    best_ppl = TrainLm_Handler().run(cmd)
    assert best_ppl < 3.0  # memorized the deterministic pattern
    assert (tmp_path / "ckpt" / "lm_best.pt").exists()


def _one_step(tmp_path, monkeypatch, grad_accum: int) -> dict[str, torch.Tensor]:
    lm = get_config().lm
    for field, value in {**_TINY_LM, "grad_accum": grad_accum}.items():
        monkeypatch.setattr(lm, field, value)
    # Pinned to CPU so the comparison is fp32. Under the production bf16 autocast the same two runs
    # agree to 2.4e-4 -- mantissa noise from regrouping the matmuls, which would swamp the property
    # being tested here (that the accumulation itself introduces no error).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tokens = np.tile(np.arange(97, dtype=np.uint16), 40)  # coprime period -> non-trivial gradients
    run_dir = tmp_path / f"accum{grad_accum}"
    run_dir.mkdir()
    (run_dir / "train.bin").write_bytes(tokens.tobytes())
    (run_dir / "val.bin").write_bytes(tokens[:200].tobytes())
    # The sampler runs off its own generator (seeded from lm.seed), so only model init reads the
    # global RNG -- seed it so both runs start from identical weights. The sampler draws indices in
    # fixed-size blocks independent of the DataLoader's batch size, so both step on the SAME 16
    # windows.
    torch.manual_seed(1234)
    TrainLm_Handler().run(
        TrainLm_Command(
            train_bin=str(run_dir / "train.bin"),
            val_bin=str(run_dir / "val.bin"),
            out_dir=str(run_dir),
            total_steps=1,
            log_dir=str(run_dir / "runs"),
        )
    )
    return torch.load(run_dir / "lm_last.pt", map_location="cpu", weights_only=False)["model"]


def _resume_fixture(tmp_path, monkeypatch):
    lm = get_config().lm
    tiny = {**_TINY_LM, "grad_accum": 1, "ckpt_every": 2, "eval_interval": 2, "branch_points": (2,)}
    for field, value in tiny.items():
        monkeypatch.setattr(lm, field, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tokens = np.tile(np.arange(97, dtype=np.uint16), 40)
    (tmp_path / "train.bin").write_bytes(tokens.tobytes())
    (tmp_path / "val.bin").write_bytes(tokens[:200].tobytes())

    def run(total_steps: int, resume: bool = True) -> dict:
        TrainLm_Handler().run(
            TrainLm_Command(
                train_bin=str(tmp_path / "train.bin"),
                val_bin=str(tmp_path / "val.bin"),
                out_dir=str(tmp_path),
                total_steps=total_steps,
                log_dir=str(tmp_path / "runs"),
                resume=resume,
            )
        )
        return torch.load(tmp_path / "lm_last.pt", map_location="cpu", weights_only=False)

    return run


def test_run_resumes_from_lm_last(tmp_path, monkeypatch):
    # A trainer that restarts at step 1 on every launch silently redoes hours of a 70k-step run
    # after a crash. This locks that lm_last.pt is picked up the way the acoustic trainers do:
    # continue the step counter (so the LR schedule stays on its curve), bump resume_count, and
    # carry the best val perplexity forward so the first post-resume eval can't overwrite
    # lm_best.pt with a worse model.
    run = _resume_fixture(tmp_path, monkeypatch)
    first = run(2)
    assert (first["step"], first["resume_count"]) == (2, 0)
    # The branch point is kept beside lm_last.pt and is a resumable checkpoint at that step.
    branch = torch.load(tmp_path / "lm_branch2.pt", map_location="cpu", weights_only=False)
    assert branch["step"] == 2 and branch["optimizers"]
    second = run(4)
    assert (second["step"], second["resume_count"]) == (4, 1)
    assert second["extra"]["val_ppl"] <= first["extra"]["val_ppl"]
    assert second["kind"] == "lm"
    # Weights moved: the resumed run trained, it did not just re-save what it loaded.
    key = "blocks.0.ffn.w_gate.weight"
    assert not torch.equal(first["model"][key], second["model"][key])


def test_fresh_ignores_lm_last(tmp_path, monkeypatch):
    # `--fresh` is the escape hatch for a config change whose old checkpoint would either fail to
    # load or silently continue a different recipe.
    run = _resume_fixture(tmp_path, monkeypatch)
    run(2)
    fresh = run(2, resume=False)
    assert (fresh["step"], fresh["resume_count"]) == (2, 0)


def test_grad_accum_is_gradient_identical(tmp_path, monkeypatch):
    # The 512-wide LM OOMs a 12 GB card at batch 64 (9.4 GB of activations), so batch_size is the
    # EFFECTIVE batch and grad_accum splits it. That split must not change the optimiser step: with
    # equal token counts per micro-batch, averaging the scaled micro-losses reproduces the
    # full-batch loss exactly, so one step must land on the same weights either way.
    full = _one_step(tmp_path, monkeypatch, grad_accum=1)
    split = _one_step(tmp_path, monkeypatch, grad_accum=2)
    assert full.keys() == split.keys()
    for name, w in full.items():
        # Tolerance covers float32 reduction-order drift between one batch-16 matmul and two
        # batch-8 ones (measured worst case 2.2e-7); the update is mathematically identical, not
        # bitwise. A real accumulation bug -- unscaled losses, a missed zero_grad, clipping inside
        # the micro loop -- moves weights by orders of magnitude more than this.
        torch.testing.assert_close(
            w, split[name], rtol=1e-5, atol=1e-7, msg=lambda m: f"{name}\n{m}"
        )
