import torch
from src.shared_kernel.Checkpoint_Adapter import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), model, opt, step=7, extra={"note": "hi"})

    fresh = torch.nn.Linear(4, 4)
    meta = load_checkpoint(str(path), fresh)
    assert meta["step"] == 7
    assert meta["extra"]["note"] == "hi"
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)
