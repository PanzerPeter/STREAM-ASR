# Export a trainer checkpoint's weights as .safetensors for public release (HuggingFace Hub).
# Training checkpoints are pickled payloads carrying optimizer state and RNG state, so they are both
# ~2x larger than the weights and unloadable without `weights_only=False`, which nobody downloading
# a model should have to accept. This writes the bare `model` state_dict in safetensors' zero-copy,
# no-pickle format, plus provenance in the file header.
import argparse
import hashlib
import os
import subprocess
from typing import Any

import torch
from safetensors.torch import save_file


def _git_commit() -> str:
    # The release's whole value is being pinned to a reproducible ref, so a dirty tree is recorded
    # as such rather than silently exported under the last clean commit's name.
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _unshare(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # safetensors stores one flat buffer per name and refuses two names pointing at one storage, so
    # tied weights (STREAM-LM ties tok_emb.weight to head.weight) abort the save. Cloning the later
    # alias costs one extra copy of a small matrix and keeps the file a complete state_dict, so the
    # consumer still loads it with strict=True and re-ties on its own if it wants to.
    seen: dict[tuple[int, torch.dtype], str] = {}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        ident = (tensor.untyped_storage().data_ptr(), tensor.dtype)
        if ident in seen:
            out[key] = tensor.clone()
        else:
            seen[ident] = key
            # contiguous() because safetensors serialises raw bytes and would otherwise write a
            # transposed view's underlying layout rather than what the tensor reads as.
            out[key] = tensor.contiguous()
    return out


def export(src: str, dst: str) -> None:
    payload: dict[str, Any] = torch.load(src, map_location="cpu", weights_only=False)
    state = _unshare(payload["model"])
    metadata = {
        # HuggingFace tooling dispatches on this key to pick the framework loader.
        "format": "pt",
        "source_checkpoint": os.path.basename(src),
        "source_sha256": _sha256(src),
        "kind": str(payload.get("kind", "")),
        "step": str(payload.get("step", 0)),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "num_tensors": str(len(state)),
        "num_parameters": str(sum(t.numel() for t in state.values())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    save_file(state, dst, metadata=metadata)
    print(f"{src} -> {dst}")
    print(f"  tensors {metadata['num_tensors']}, params {int(metadata['num_parameters'])/1e6:.2f}M")
    print(f"  {os.path.getsize(dst)/1e6:.1f} MB, commit {metadata['git_commit']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="export checkpoint weights to .safetensors")
    ap.add_argument("--src", required=True, help="input .pt checkpoint")
    ap.add_argument("--dst", required=True, help="output .safetensors path")
    args = ap.parse_args()
    export(args.src, args.dst)


if __name__ == "__main__":
    main()
