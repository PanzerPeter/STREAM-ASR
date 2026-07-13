import os
import random

import torch

from src.shared_kernel.Checkpoint_Adapter import save_checkpoint, load_checkpoint


def _model_opt():
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_roundtrip_restores_step_best_and_optimizer(tmp_path):
    model, opt = _model_opt()
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model, opt, step=42, best_wer=0.13, resume_count=1, kind="stage_a")

    model2, opt2 = _model_opt()
    meta = load_checkpoint(path, model2, opt2)
    assert meta["step"] == 42
    assert meta["best_wer"] == 0.13
    assert meta["resume_count"] == 1
    assert torch.allclose(model.weight, model2.weight)
    assert opt2.state_dict()["state"]  # optimizer moments restored


def test_optimizer_list_roundtrips(tmp_path):
    model = torch.nn.Linear(4, 3)
    o1 = torch.optim.SGD([model.weight], lr=0.1)
    o2 = torch.optim.AdamW([model.bias], lr=0.1)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model, [o1, o2], step=1)
    n1 = torch.optim.SGD([model.weight], lr=0.1)
    n2 = torch.optim.AdamW([model.bias], lr=0.1)
    load_checkpoint(path, model, [n1, n2])  # must not raise on the two-optimizer list


def test_rng_state_restored(tmp_path):
    model, opt = _model_opt()
    path = str(tmp_path / "ck.pt")
    random.seed(0)
    torch.manual_seed(0)
    save_checkpoint(path, model, opt, step=0)
    expected = (random.random(), torch.rand(1).item())
    random.seed(999)
    torch.manual_seed(999)
    load_checkpoint(path, model, opt)  # restores RNG captured at save time
    assert (random.random(), torch.rand(1).item()) == expected


def test_atomic_no_partial_on_replace(tmp_path):
    model, opt = _model_opt()
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model, opt, step=1)
    save_checkpoint(path, model, opt, step=2)
    assert not os.path.exists(path + ".tmp")  # tmp cleaned by os.replace
    assert load_checkpoint(path, model, opt)["step"] == 2
