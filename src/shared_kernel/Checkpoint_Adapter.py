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


def average_checkpoints(paths: list[str], out_path: str, map_location: str = "cpu") -> str:
    # Checkpoint averaging (icefall/ESPnet): mean the model weights of several late-training
    # snapshots into one checkpoint. Averaging points along the SGD trajectory approximates a wider,
    # flatter minimum and reliably shaves ASR WER at zero training cost. Accumulate in float64 so a
    # long tail of bf16-trained snapshots does not lose precision, then cast back to each tensor's
    # dtype. Non-float buffers (integer counters) are undefined under a mean, so the first
    # snapshot's value is kept. The output payload is shaped like save_checkpoint's, so
    # load_checkpoint reads it unchanged (empty optimizers, no RNG to restore).
    if not paths:
        raise ValueError("average_checkpoints needs at least one checkpoint path")
    states = [torch.load(p, map_location=map_location, weights_only=False)["model"] for p in paths]
    n = len(states)
    avg: dict = {}
    for key, ref in states[0].items():
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for s in states:
                acc += s[key].to(torch.float64)
            avg[key] = (acc / n).to(ref.dtype)
        else:
            avg[key] = ref.clone()
    payload = {
        "model": avg,
        "optimizers": [],
        "step": 0,
        "best_wer": math.inf,
        "resume_count": 0,
        "kind": "transducer-avg",
        "rng": None,
        "extra": {"averaged_from": list(paths)},
    }
    tmp = out_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    return out_path


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
