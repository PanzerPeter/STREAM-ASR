# src/shared_kernel/Checkpoint_Adapter.py
import math
import os
import random

import torch


def _as_list(optimizers):
    if optimizers is None:
        return []
    return list(optimizers) if isinstance(optimizers, (list, tuple)) else [optimizers]


def save_checkpoint(
    path: str,
    model,
    optimizers,
    step: int,
    *,
    best_wer: float = math.inf,
    resume_count: int = 0,
    kind: str = "",
    extra: dict | None = None,
) -> None:
    # Atomic write: torch.save to a sibling .tmp then os.replace (a same-dir rename is atomic on
    # POSIX), so a process killed mid-write never truncates the live checkpoint — at worst it leaves
    # an orphan .tmp. This is the core of the SIGINT-safe harness (SP2).
    payload = {
        "model": model.state_dict(),
        "optimizers": [o.state_dict() for o in _as_list(optimizers)],
        "step": step,
        "best_wer": best_wer,
        "resume_count": resume_count,
        "kind": kind,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optimizers=None, map_location="cpu") -> dict:
    # weights_only=False: the payload carries python/torch RNG state (arbitrary pickled objects),
    # which the torch 2.11 weights_only default would reject.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opts = _as_list(optimizers)
    for opt, state in zip(opts, ckpt["optimizers"]):
        opt.load_state_dict(state)
    rng = ckpt.get("rng")
    if rng is not None:
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch_cpu"])
        if rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return {
        "step": ckpt["step"],
        "best_wer": ckpt.get("best_wer", math.inf),
        "resume_count": ckpt.get("resume_count", 0),
        "extra": ckpt.get("extra", {}),
    }


def resume_if_available(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    resume: bool,
) -> dict[str, float]:
    # Pragmatic resume (SP2): restore weights/optimizer(s)/step/best_wer/RNG, and bump resume_count
    # so the caller can reseed a *fresh* shuffled epoch (base_seed + resume_count) rather than
    # replay the exact pre-interrupt batch order — indistinguishable in final WER over a long run.
    if not resume or not os.path.isfile(ckpt_path):
        return {"step": 0, "best_wer": math.inf, "resume_count": 0}
    meta = load_checkpoint(ckpt_path, model, optimizers)
    return {
        "step": meta["step"],
        "best_wer": meta["best_wer"],
        "resume_count": meta["resume_count"] + 1,
    }
