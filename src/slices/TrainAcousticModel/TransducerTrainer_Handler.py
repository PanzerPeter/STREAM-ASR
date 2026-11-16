# The joint transducer stage: S2 encoder + CTC head + predictor + joiner under one objective, from
# scratch. Runs on the resumable, SIGINT-safe harness (Checkpoint_Adapter + SignalGuard)
# and selects checkpoints on the head that ships: full dev-other streaming greedy-transducer WER.
import os
import random
import time
from datetime import timedelta

import torch
import torch._dynamo
from torch.utils.tensorboard import SummaryWriter

from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.GradientClipping import (
    clip_grads_per_tensor,
    grad_norm_of,
    guarded_parameters,
    unguarded_parameters,
)
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.LrSchedule import lr_at
from src.shared_kernel.Optimizer_Adapter import build_optimizer, partition_params
from src.shared_kernel.SignalGuard import SignalGuard
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.DevMetrics import dev_metrics
from src.slices.TrainAcousticModel.FusedBranchBlock import FusedBranchBlock
from src.slices.TrainAcousticModel.GradNormGuard import GradNormGuard
from src.slices.TrainAcousticModel.SnapshotRotation import (
    drop_snapshots_after,
    write_branch_point,
    write_rolling_snapshot,
)
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TrainingHealth import health_scalars
from src.slices.TrainAcousticModel.TrainingLoaders import build_loaders
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand

# Checkpoint selection metric (MODEL_ARCHITECTURE.md §9.6).
_SELECT = "transducer_wer_stream_other"
# dynamo's recompile limit is per code object and one FusedBranchBlock.forward serves every block:
# widths 256/384 x mask/no-mask x grad modes x dynamic-shape variants. Exceeding it does not raise,
# it silently demotes the block to eager for the rest of the process.
_RECOMPILE_LIMIT = 64


def _require_cmvn(cmvn_path: str) -> str:
    # The encoder's mean 0 / std 1 fallback is right for tests and inference (a checkpoint carries
    # the real buffers) and silently wrong for a training run.
    if not os.path.isfile(cmvn_path):
        raise FileNotFoundError(f"cmvn statistics not found: {cmvn_path}; run compute_cmvn.py")
    return cmvn_path


def _compile_blocks(model: TransducerModel) -> int:
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, _RECOMPILE_LIMIT
    )
    blocks = [m for m in model.modules() if isinstance(m, FusedBranchBlock)]
    for block in blocks:
        # nn.Module.compile keeps state_dict keys, so checkpoints load into the eager decode path.
        block.compile(dynamic=True)
    return len(blocks)


def run_transducer(cmd: TransducerTrainCommand) -> str:
    # The lattice is a large size-varying allocation; expandable_segments stops the default caching
    # allocator fragmenting around it. Read at the first CUDA allocation, so set before it.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = configure_logging()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tr = get_config().training.transducer
    random.seed(tr.seed)
    torch.manual_seed(tr.seed)
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    model = TransducerModel(cmvn_path=_require_cmvn(cmd.cmvn_path)).to(device)
    train_loader, sampler, dev_sets = build_loaders(cmd, tokenizer)
    probe = next(iter(dev_sets[-1].loader))
    n_compiled = _compile_blocks(model) if tr.compile_blocks else 0
    optimizers = build_optimizer(model, get_config().optim)
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    muon_params, _ = partition_params(model)
    guard_params, gate_params = guarded_parameters(model), unguarded_parameters(model)
    t_cfg = get_config().transducer
    log.info(
        f"STREAM ASR · Transducer │ {device} · params "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M · steps {cmd.total_steps:,} · "
        f"band W={t_cfg.band_width} after {tr.band_warmup_steps:,} · "
        f"compiled blocks {n_compiled or 'off'}"
    )

    last_ckpt = os.path.join(cmd.ckpt_dir, "transducer_last.pt")
    best_ckpt = os.path.join(cmd.ckpt_dir, "transducer_best.pt")
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step, best_wer = int(resumed["step"]), float(resumed["best_wer"])
    resume_count = int(resumed["resume_count"])
    guard = GradNormGuard(
        tr.guard_window,
        tr.guard_trend_factor,
        tr.guard_patience,
        floor=float(resumed["extra"].get("guard_norm_floor", 0.0)),
    )
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
        drop_snapshots_after(cmd.ckpt_dir, step, log)
    sampler._seed = tr.seed + resume_count
    writer = SummaryWriter(cmd.log_dir)

    def save(path: str) -> None:
        save_checkpoint(
            path,
            model,
            optimizers,
            step,
            best_wer=best_wer,
            resume_count=resume_count,
            kind="transducer",
            extra={"guard_norm_floor": guard.floor},
        )

    def validate() -> None:
        nonlocal best_wer
        metrics = dev_metrics(model, dev_sets, probe, tokenizer, device)
        for name, val in metrics.items():
            # Gate saturation is a property of the model, measured on a dev batch only because the
            # compiled blocks cannot export activations mid-training.
            writer.add_scalar(f"{'train' if name == 'gate_sat_frac' else 'dev'}/{name}", val, step)
        best = metrics[_SELECT] < best_wer
        best_wer = min(best_wer, metrics[_SELECT])
        if best:
            save(best_ckpt)
        if metrics["band_loss_gap"] > 0.01:
            log.warning(
                f"dev/band_loss_gap {metrics['band_loss_gap']:.4f} nats/token > 0.01: the band "
                "is dropping mass; widen band_width or the warmup."
            )
        log.log(
            "SUCCESS" if best else "INFO",
            " │ ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            + f" @ step {step:,}{'  ← best' if best else ''}",
        )

    run_start, start_step = time.perf_counter(), step
    win_start, win_step = run_start, step
    norms = (torch.zeros((), device=device), torch.zeros((), device=device))
    accumulated = oom_skips = 0
    model.train()
    with SignalGuard() as stop:
        while step < cmd.total_steps:
            for batch in train_loader:
                shape = lr_at(
                    step,
                    1.0,
                    tr.warmup_steps,
                    cmd.total_steps,
                    decay_frac=tr.lr_decay_frac,
                    min_ratio=tr.lr_min_ratio,
                )
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * shape
                # One (chunk, left) pair per batch, so every batch is one mask shape.
                chunk = random.choice(tr.chunk_sizes)
                left = random.choice(tr.left_contexts)
                try:
                    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
                        terms = model.joint_loss(batch, chunk, left, step >= tr.band_warmup_steps)
                    (terms.total / tr.grad_accum).backward()
                except torch.OutOfMemoryError:
                    # One dense bucket failing is transient; the partial backward left gradients
                    # on only some parameters, so the whole accumulation window goes.
                    oom_skips, accumulated = oom_skips + 1, 0
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    del batch
                    torch.cuda.empty_cache()
                    log.warning(f"CUDA OOM @ step {step:,}: batch dropped (skips {oom_skips})")
                    continue

                accumulated += 1
                if accumulated == tr.grad_accum:
                    # Guard reads the matrices BEFORE their clip; scalars and vectors are clipped
                    # per tensor so no one gate sets everyone's clip factor.
                    norms = (
                        grad_norm_of(guard_params),
                        clip_grads_per_tensor(gate_params, tr.grad_clip),
                    )
                    torch.nn.utils.clip_grad_norm_(guard_params, tr.grad_clip)
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    accumulated = 0

                if step % tr.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    scalars = {
                        "loss": terms.total,
                        "rnnt": terms.rnnt,
                        "ctc": terms.ctc,
                        "band_scale": terms.band_scale,
                        "band_infeasible": terms.band_infeasible,
                        "grad_norm_guarded": norms[0],
                        "grad_norm": norms[1],
                        **health_scalars(model, muon_params),
                    }
                    values = torch.stack([v.detach().float() for v in scalars.values()]).tolist()
                    logged = dict(zip(scalars, values))
                    logged["lr"] = optimizers[-1].param_groups[0]["lr"]
                    for name, val in logged.items():
                        writer.add_scalar(name if "/" in name else f"train/{name}", val, step)
                    eta = timedelta(seconds=int((cmd.total_steps - step) / its)) if its else "-"
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} │ loss {logged['loss']:6.3f} "
                        f"(rnnt {logged['rnnt']:5.2f} ctc {logged['ctc']:5.2f}) │ "
                        f"chunk {chunk} left {left} │ "
                        f"lr {logged['lr']:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    if logged["band_infeasible"] > 0:
                        log.error(f"{logged['band_infeasible']:.0f} band(s) miss a terminal")
                    abort, median = (
                        guard.update(logged["grad_norm_guarded"])
                        if step >= tr.warmup_steps
                        else (False, 0.0)
                    )
                    if abort:
                        log.error(
                            f"GRADIENT RUNAWAY @ step {step:,}: weight-matrix grad-norm median "
                            f"{median:.2f} vs floor {guard.floor:.2f}. NOT checkpointing; resume "
                            "from transducer_last.pt or a transducer_step*.pt that predates it."
                        )
                        writer.close()
                        return last_ckpt
                    win_start, win_step = now, step

                if step > 0 and step % tr.val_every == 0:
                    validate()
                if step > 0 and step % tr.ckpt_every == 0:
                    save(last_ckpt)
                    write_rolling_snapshot(cmd.ckpt_dir, last_ckpt, step, tr.keep_last_n)
                if step in tr.branch_points:
                    save(last_ckpt)
                    write_branch_point(cmd.ckpt_dir, last_ckpt, step)

                step += 1
                if stop.stop_requested:
                    save(last_ckpt)
                    log.warning(f"interrupt received; checkpointed @ step {step:,}, exiting.")
                    writer.close()
                    return last_ckpt
                if step >= cmd.total_steps:
                    break

    save(last_ckpt)
    # The loop validates before a step's increment, so the final weights would otherwise never be
    # scored; a cooldown branch is read off exactly this line.
    validate()
    writer.close()
    wall = time.perf_counter() - run_start
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    log.success(
        f"done │ steps {step:,} · {timedelta(seconds=int(wall))} · "
        f"{wall / max(1, step - start_step):.3f} s/step this process · peak {peak:.2f} GiB · "
        f"best {_SELECT} {best_wer:.4f}"
    )
    return last_ckpt
