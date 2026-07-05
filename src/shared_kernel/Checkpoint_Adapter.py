# src/shared_kernel/Checkpoint_Adapter.py
import torch


def save_checkpoint(path: str, model, optimizer, step: int, extra: dict | None = None) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"step": ckpt["step"], "extra": ckpt["extra"]}
