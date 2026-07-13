# src/slices/TrainAcousticModel/StageATrainer_Handler.py
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import jiwer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.shared_kernel.Checkpoint_Adapter import save_checkpoint, resume_if_available
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.SignalGuard import SignalGuard
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.AcousticModel import AcousticModel
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand


def load_pretrained_encoder(model: AcousticModel, path: str) -> None:
    # Warm-start the acoustic encoder from a BEST-RQ pretrain checkpoint (SP4). The checkpoint
    # holds exactly encoder.* weights; loading with strict=True guarantees a shape/name match and
    # fails loud on drift. The CTC head stays randomly initialized.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["model"], strict=True)


def _lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)  # cosine decay to 0
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _seed_all(seed: int) -> None:
    # Seed model init, worker augmentation, and batch order so the blank-collapse escape (an init-
    # sensitive knife-edge) is reproducible. torch.manual_seed also fixes the DataLoader workers'
    # per-worker seeds (PyTorch derives them from the main generator), so SpecAugment/speed-perturb
    # become deterministic too. use_deterministic_algorithms is deliberately NOT set: cuDNN's CTC
    # has no deterministic kernel and would raise.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _startup_table(
    cmd: StageATrainCommand,
    device: str,
    n_params: int,
    n_train: int,
    n_dev: int,
    checkpointed: bool,
) -> Table:
    sa = get_config().training.stage_a
    optim = get_config().optim
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("k", style="cyan", justify="right")
    t.add_column("v", style="white")
    precision = "bf16 autocast" if device == "cuda" else "fp32"
    if cmd.compile_model:
        accel = "torch.compile"
    else:
        accel = "eager + grad-checkpoint" if checkpointed else "eager"
    rows = [
        ("device", device),
        ("params", f"{n_params / 1e6:.1f} M"),
        ("train / dev utts", f"{n_train:,} / {n_dev:,}"),
        ("total steps", f"{cmd.total_steps:,}"),
        ("max frames / batch", f"{sa.max_frames_per_batch:,}"),
        ("grad accum", str(sa.grad_accum)),
        ("lr peak muon/adamw", f"{optim.muon_lr:g} / {optim.adamw_lr:g}"),
        ("warmup steps", f"{sa.warmup_steps:,}"),
        ("precision", precision),
        ("execution", accel),
        ("checkpoints → ", cmd.ckpt_dir),
        ("tensorboard → ", cmd.log_dir),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


@torch.no_grad()
def _dev_wer(model, loader, tokenizer, device) -> tuple[float, float, list[tuple[str, str]]]:
    """Returns (WER, mean blank-argmax fraction, a few (ref, hyp) samples). The blank fraction
    and samples make the early-CTC blank plateau observable: WER sits pinned at 1.0 while the
    model emits blank everywhere, so blank_frac falling from ~1.0 — and hyps turning non-empty —
    is the earliest signal that alignment is forming, well before WER moves."""
    blank_id = get_config().model.blank_id
    model.eval()
    refs, hyps = [], []
    blank_frames = 0
    total_frames = 0
    samples: list[tuple[str, str]] = []
    for batch in loader:
        logits, out_len = model(batch.features.to(device), batch.feature_lengths.to(device))
        best = logits.float().cpu().argmax(dim=-1)  # [B, T]
        for b in range(best.shape[0]):
            valid = int(out_len[b])
            blank_frames += int((best[b, :valid] == blank_id).sum())
            total_frames += valid
        batch_hyps = ctc_greedy_decode(logits.float().cpu(), out_len.cpu(), tokenizer)
        hyps.extend(batch_hyps)
        for i in range(batch.tokens.shape[0]):
            ref = tokenizer.decode(batch.tokens[i, : batch.token_lengths[i]].tolist())
            refs.append(ref)
            if len(samples) < 3:
                samples.append((ref, batch_hyps[i]))
    model.train()
    blank_frac = blank_frames / max(total_frames, 1)
    return jiwer.wer(refs, hyps), blank_frac, samples


class _Checkpointed(torch.nn.Module):
    """Wraps a stack so its forward runs under activation checkpointing."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, lengths, pad_mask, chunk_size=0):
        return torch.utils.checkpoint.checkpoint(
            self.module, x, lengths, pad_mask, chunk_size, use_reentrant=False
        )


def run_stage_a(cmd: StageATrainCommand) -> str:
    log = configure_logging()
    console = Console()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    if cmd.device == "cuda" and device == "cpu":
        log.warning("CUDA requested but unavailable — falling back to CPU (training will be slow).")
    # TF32 tensor-core path for the fp32 ops outside bf16 autocast (CTC log-softmax, norms).
    torch.set_float32_matmul_precision("high")
    log.info(f"Stage-A CTC training on <{device}>")
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    writer = SummaryWriter(cmd.log_dir)
    sa = get_config().training.stage_a
    _seed_all(sa.seed)
    log.info(f"seed {sa.seed} (reproducible init/augment/batch order)")

    train_ds = LibriSpeechDataset(cmd.train_manifest, tokenizer, train=True)
    train_sampler = FrameBucketSampler(
        cmd.train_manifest, sa.max_frames_per_batch, shuffle=True, seed=sa.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_features,
        num_workers=4,
        pin_memory=True,
    )

    dev_ds = LibriSpeechDataset(cmd.dev_manifest, tokenizer, train=False)
    dev_sampler = FrameBucketSampler(cmd.dev_manifest, sa.max_frames_per_batch)
    dev_loader = DataLoader(dev_ds, batch_sampler=dev_sampler, collate_fn=collate_features)

    cmvn = cmd.cmvn_path if os.path.isfile(cmd.cmvn_path) else None
    model = AcousticModel(cmvn_path=cmvn).to(device)

    if cmd.encoder_init is not None and os.path.isfile(cmd.encoder_init):
        load_pretrained_encoder(model, cmd.encoder_init)
        log.info(f"warm-started encoder from {cmd.encoder_init}")
    elif cmd.encoder_init is not None:
        log.warning(f"encoder_init {cmd.encoder_init} not found — training from random init")

    # torch.compile rematerializes activations via its own partitioner, so checkpointing only
    # applies to the eager path, and is skipped unless config asks for it (VRAM has ~7 GB of
    # headroom at 20k frames, and checkpointing costs ~30% step time — see grad_checkpoint).
    checkpointed = not cmd.compile_model and sa.grad_checkpoint
    if cmd.compile_model:
        # Inductor's min-cut partitioner rematerializes activations itself, so we do NOT also
        # wrap stacks in torch.utils.checkpoint — that combo breaks the partitioner under bf16.
        # dynamic=True compiles once for variable-length batches; max-autotune is skipped because
        # per-shape autotune recompiles are impractical for dynamic ASR batch shapes.
        forward = torch.compile(model, dynamic=True)
    else:
        if checkpointed:
            # Bound activation memory by recomputing each stack's forward in the backward pass.
            model.encoder.stacks = torch.nn.ModuleList(
                [_Checkpointed(s) for s in model.encoder.stacks]
            )
        forward = model
    optimizers = build_optimizer(model, get_config().optim)
    # build_optimizer sets each group's lr to its calibrated PEAK (Muon >> AdamW per
    # config/optim.yaml, incl. any muP per-group scaling). Snapshot the peaks so the warmup+cosine
    # schedule is applied as a 0->1->0 SHAPE multiplier per group. A single absolute overwrite would
    # clobber Muon's much larger base LR and any muP ratios, defeating SP3. NB: optimizer peak LRs
    # now come from optim.yaml (adamw_lr / muon_lr), so the AdamW peak = adamw_lr.
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    n_params = sum(p.numel() for p in model.parameters())
    console.print(
        Panel(
            _startup_table(cmd, device, n_params, len(train_ds), len(dev_ds), checkpointed),
            title="[bold]STREAM · Stage-A[/bold]",
            border_style="cyan",
            expand=False,
        )
    )

    last_ckpt = os.path.join(cmd.ckpt_dir, "stage_a_last.pt")
    best_ckpt = os.path.join(cmd.ckpt_dir, "stage_a_best.pt")
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step = int(resumed["step"])
    best_wer = resumed["best_wer"]
    resume_count = int(resumed["resume_count"])
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
    # Reseed the sampler for a fresh, non-repeating epoch after a resume (read at iteration time
    # by FrameBucketSampler.__iter__, so setting it here before the loop starts takes effect).
    train_sampler._seed = sa.seed + resume_count
    run_start = time.perf_counter()
    win_start, win_step = run_start, 0  # throughput/ETA window
    win_loss = torch.zeros((), device=device)  # accumulate on-device; sync only at log_every
    model.train()
    log.info(f"Training loop started — target {cmd.total_steps:,} steps.")
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for batch in train_loader:
                lr_shape = _lr_at(step, 1.0, sa.warmup_steps, cmd.total_steps)  # 0->1->0 shape
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                # Representative LR for logging/tensorboard: build_optimizer returns AdamW last
                # ([muon, adamw] or [adamw]), so optimizers[-1] is always the AdamW.
                lr = optimizers[-1].param_groups[0]["lr"]

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    logits, out_len = forward(
                        batch.features.to(device, non_blocking=True),
                        batch.feature_lengths.to(device, non_blocking=True),
                    )
                    loss = (
                        model.ctc_loss(
                            logits,
                            out_len,
                            batch.tokens.to(device, non_blocking=True),
                            batch.token_lengths.to(device, non_blocking=True),
                        )
                        / sa.grad_accum
                    )
                loss.backward()

                if (step + 1) % sa.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), sa.grad_clip)
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                win_loss += loss.detach() * sa.grad_accum  # kept on-device; no per-step sync

                if step % sa.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    avg_loss = (win_loss / max(step - win_step, 1)).item()  # single sync per window
                    # its is 0 on the first log (no window elapsed yet) — show a placeholder ETA
                    # rather than the step/0 → 33333333333:20:00 artifact.
                    eta = _fmt_hms((cmd.total_steps - step) / its) if its > 0 else "—"
                    pct = 100.0 * step / cmd.total_steps
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} ({pct:4.1f}%) │ "
                        f"loss {avg_loss:7.3f} │ lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/it_per_s", its, step)
                    win_start, win_step = now, step
                    win_loss.zero_()
                if step > 0 and step % sa.val_every == 0:
                    wer, blank_frac, samples = _dev_wer(model, dev_loader, tokenizer, device)
                    writer.add_scalar("dev/wer", wer, step)
                    writer.add_scalar("dev/blank_frac", blank_frac, step)
                    best = wer < best_wer
                    best_wer = min(best_wer, wer)
                    marker = "  ← best" if best else f"  (best {best_wer:.4f})"
                    log.log(
                        "SUCCESS" if best else "INFO",
                        f"dev WER {wer:.4f}{marker} │ blank {blank_frac:.3f}  @ step {step:,}",
                    )
                    # Persist the best-WER weights separately: `..._last.pt` can drift/overfit
                    # past the optimum over a long run, so the shippable checkpoint is this one.
                    if best:
                        save_checkpoint(
                            best_ckpt,
                            model,
                            optimizers,
                            step,
                            best_wer=best_wer,
                            resume_count=resume_count,
                            kind="stage_a",
                        )
                    # Sample decodes surface the phase transition before WER numerically moves.
                    for ref, hyp in samples:
                        log.debug(f"  ref: {ref[:80]!r}")
                        log.debug(f"  hyp: {hyp[:80]!r}")
                    # Fast-fail the blank-collapse trap, but only when BOTH signals agree the run
                    # is dead: blank still collapsed AND dev WER never fell below
                    # escape_min_wer_progress. The escape onset is noisy — blank_frac can read
                    # ~1.0 at the check step while WER has already dipped off 1.0 (alignment
                    # forming) — so keying on blank_frac alone guillotines runs that are escaping
                    # the saddle slower than escape_check_step.
                    collapsed = blank_frac > sa.escape_max_blank_frac
                    no_progress = best_wer >= sa.escape_min_wer_progress
                    if step >= sa.escape_check_step and collapsed and no_progress:
                        writer.close()
                        raise RuntimeError(
                            f"CTC blank collapse: blank_frac {blank_frac:.3f} > "
                            f"{sa.escape_max_blank_frac} and best dev WER {best_wer:.3f} >= "
                            f"{sa.escape_min_wer_progress} at step {step:,} (> "
                            f"escape_check_step {sa.escape_check_step:,}). Model never left "
                            f"the all-blank saddle — raise optim.yaml adamw_lr/muon_lr / lengthen "
                            f"warmup or add the Zipformer stabilizers, then restart."
                        )
                if step > 0 and step % sa.ckpt_every == 0:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        best_wer=best_wer,
                        resume_count=resume_count,
                        kind="stage_a",
                    )
                    log.debug(f"checkpoint saved → {last_ckpt}  @ step {step:,}")

                step += 1
                if guard.stop_requested:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        best_wer=best_wer,
                        resume_count=resume_count,
                        kind="stage_a",
                    )
                    log.warning(f"interrupt received — checkpointed @ step {step:,}; exiting.")
                    writer.close()
                    return last_ckpt
                if step >= cmd.total_steps:
                    break

    save_checkpoint(
        last_ckpt,
        model,
        optimizers,
        step,
        best_wer=best_wer,
        resume_count=resume_count,
        kind="stage_a",
    )
    writer.close()
    elapsed = _fmt_hms(time.perf_counter() - run_start)
    console.print(
        Panel(
            f"steps {step:,} · elapsed {elapsed} · best dev WER "
            f"{best_wer if math.isfinite(best_wer) else float('nan'):.4f}\n"
            f"last → {last_ckpt}\nbest → {best_ckpt}",
            title="[bold green]Stage-A complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    return last_ckpt
