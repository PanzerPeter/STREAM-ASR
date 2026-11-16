# Where a transducer training step spends its time (GPU; user-run): the real loader, budgets and
# compile setting, with the band on (the steady state) and off (the band warmup), plus peak VRAM
# and a kernel table (MODEL_ARCHITECTURE.md §9.4). Run it before and after any speed change, and
# on a new GPU before sizing the batch budgets.
import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.TrainingLoaders import build_loaders
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import _compile_blocks

# Inductor builds its graphs lazily over the first distinct shape classes; steps inside that window
# would charge compile time to the step.
_WARMUP_COMPILED = 10
_WARMUP_EAGER = 2


def _step(model: TransducerModel, batch, optimizers, use_band: bool) -> None:
    with torch.autocast("cuda", torch.bfloat16):
        terms = model.joint_loss(batch, 8, 128, use_band)
    terms.total.backward()
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)


def _timed(model, loader, optimizers, steps: int, warmup: int, use_band: bool) -> None:
    it = iter(loader)
    for _ in range(warmup):
        _step(model, next(it), optimizers, use_band)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    frames, start = 0, time.perf_counter()
    for _ in range(steps):
        batch = next(it)
        frames += int(batch.feature_lengths.sum())
        _step(model, batch, optimizers, use_band)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1e3 / steps
    a = get_config().audio
    rt = frames * a.hop_length / a.sample_rate / (ms / 1e3 * steps)
    gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"  band {'on ' if use_band else 'off'} │ {ms:6.1f} ms/step │ {rt:6.0f}x realtime │ "
        f"peak {gib:.2f} GiB"
    )


def main() -> None:
    # The trainer's allocator: without it peak VRAM reads fragmentation, not footprint.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Profile one transducer training step.")
    p.add_argument("--steps", type=int, default=20, help="timed steps after the warmup")
    p.add_argument("--profile-steps", type=int, default=4, help="steps under torch.profiler")
    p.add_argument("--no-compile", action="store_true", help="eager, regardless of config")
    a = p.parse_args()

    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    model = TransducerModel(cmvn_path=cmd.cmvn_path).cuda().train()
    loader, _, _ = build_loaders(cmd, tokenizer)
    compiled = get_config().training.transducer.compile_blocks and not a.no_compile
    if compiled:
        _compile_blocks(model)
    optimizers = build_optimizer(model, get_config().optim)
    print(
        f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params · compiled {compiled}"
    )

    it, start, seen = iter(loader), time.perf_counter(), 0
    for _ in range(30):
        seen += int(next(it).feature_lengths.sum())
    au = get_config().audio
    supply = seen * au.hop_length / au.sample_rate / (time.perf_counter() - start)
    print(f"  loader supplies {supply:.0f}x realtime of distinct audio")

    warmup = _WARMUP_COMPILED if compiled else _WARMUP_EAGER
    for use_band in (False, True):
        _timed(model, loader, optimizers, a.steps, warmup, use_band)

    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.profile_steps):
            _step(model, next(it), optimizers, True)
        torch.cuda.synchronize()
    avgs = prof.key_averages()
    launches = sum(e.count for e in avgs if e.self_device_time_total > 0)
    print(f"\n  {launches / a.profile_steps:.0f} kernel launches/step (band on)")
    print(avgs.table(sort_by="self_device_time_total", row_limit=25))


if __name__ == "__main__":
    main()
