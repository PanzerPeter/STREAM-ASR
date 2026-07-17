# tests/slices/test_trainer_resume.py
import torch

from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint


def _model_opt():
    m = torch.nn.Linear(4, 3)
    return m, torch.optim.AdamW(m.parameters(), lr=1e-3)


def test_fresh_run_when_no_checkpoint(tmp_path):
    m, o = _model_opt()
    meta = resume_if_available(str(tmp_path / "nope.pt"), m, [o], resume=True)
    assert meta == {"step": 0, "best_wer": float("inf"), "resume_count": 0}


def test_resumes_and_bumps_resume_count(tmp_path):
    m, o = _model_opt()
    path = str(tmp_path / "last.pt")
    save_checkpoint(path, m, [o], step=500, best_wer=0.2, resume_count=0, kind="stage_a")
    m2, o2 = _model_opt()
    meta = resume_if_available(path, m2, [o2], resume=True)
    assert meta["step"] == 500
    assert meta["best_wer"] == 0.2
    assert meta["resume_count"] == 1  # incremented for the fresh post-resume epoch seed


def test_resume_false_ignores_checkpoint(tmp_path):
    m, o = _model_opt()
    path = str(tmp_path / "last.pt")
    save_checkpoint(path, m, [o], step=500, kind="stage_a")
    m2, o2 = _model_opt()
    meta = resume_if_available(path, m2, [o2], resume=False)
    assert meta["step"] == 0
