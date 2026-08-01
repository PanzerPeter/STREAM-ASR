# STREAM-LM training: Muon+AdamW + warmup->cosine, bf16 autocast (eager), z-loss, val perplexity,
# best-ckpt. Windows are document-masked, so a training position only ever attends its own corpus
# line -- the same context a rescored ASR hypothesis has at decode time.
# Terminal logging mirrors the acoustic trainers (Logging_Adapter loguru sink + rich Panels +
# TensorBoard) so a multi-hour LM run is monitored the same way.
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.panel import Panel

from src.shared_kernel.Config_Adapter import LmConfig, get_config
from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Muon_Optimizer import Muon
from src.shared_kernel.Optimizer_Adapter import partition_params
from src.shared_kernel.SignalGuard import SignalGuard
from src.slices.TrainLanguageModel.LmDataset import LmDataset
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command

# Windows scored per validation pass. Bounded so a mid-run eval costs seconds, not minutes.
_VAL_WINDOWS = 1280


def _fmt_hms(seconds: float) -> str:
    # Duplicated from the acoustic trainers' identical helper rather than imported: AC-002 forbids
    # reaching into another slice's internals, and VSA makes duplication (not a shared-kernel
    # promotion) the default until an explicit /abstract command.
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class TrainLm_Handler:
    def run(self, cmd: TrainLm_Command) -> float:
        # Same allocator setting the transducer trainer uses: micro-batch activation blocks vary in
        # size across the run, and growing a segment in place beats rounding up to a fresh block.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = configure_logging()
        console = Console()
        lm = get_config().lm
        if lm.batch_size % lm.grad_accum:
            raise ValueError(
                f"lm.batch_size {lm.batch_size} not divisible by lm.grad_accum {lm.grad_accum}"
            )
        micro = lm.batch_size // lm.grad_accum
        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp = device == "cuda"
        torch.set_float32_matmul_precision("high")
        log.info(f"STREAM-LM training on <{device}>")

        model = StreamLmModel().to(device)
        optimizers = self._optimizers(model, lm)
        # Peak LR per group, captured once: the schedule below is a SHAPE in [0, 1] that multiplies
        # them, so Muon's and AdamW's very different peaks keep their ratio through warmup + decay.
        peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
        steps = min(cmd.max_steps, lm.total_steps)
        Path(cmd.out_dir).mkdir(parents=True, exist_ok=True)
        best_ckpt = f"{cmd.out_dir}/lm_best.pt"
        last_ckpt = f"{cmd.out_dir}/lm_last.pt"

        # Restore model + both optimizers + RNG and pick the step counter back up, exactly as the
        # acoustic trainers do. `val_ppl` is this trainer's selection metric and comes back with it,
        # so a resumed run overwrites lm_best.pt only when it beats the pre-interrupt best.
        resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
        start = int(resumed["step"])
        resume_count = int(resumed["resume_count"])
        best = float(resumed["extra"].get("val_ppl", math.inf))
        if start > 0:
            log.info(
                f"resumed from {last_ckpt} @ step {start:,} (resume #{resume_count}, "
                f"best val ppl {best:.3f})"
            )
        # Sampler seed carries resume_count so a restart draws a fresh stretch of the window stream
        # instead of re-walking the windows the interrupted run already trained on.
        loader = self._loader(cmd.train_bin, lm.context_len, micro, lm.seed + resume_count)
        val = LmDataset(cmd.val_bin, lm.context_len)
        writer = SummaryWriter(cmd.log_dir)

        n_params = sum(p.numel() for p in model.parameters())
        console.print(
            Panel(
                f"device {device} · params {n_params / 1e6:.1f} M · steps {start:,} → {steps:,}\n"
                f"d_model {lm.d_model} · layers {lm.layers} · heads {lm.heads} "
                f"(kv {lm.kv_groups}) · ctx {lm.context_len}\n"
                f"batch {lm.batch_size} ({lm.grad_accum} x {micro}) · {lm.optimizer} · "
                f"adamw lr {lm.lr_peak:g} / muon lr "
                f"{lm.muon_lr:g} / warmup {lm.warmup_steps:,} · eval every {lm.eval_interval:,}\n"
                f"resume {'#' + str(resume_count) if start else 'fresh'} · "
                f"ckpt every {lm.ckpt_every:,} → {cmd.out_dir} · tb → {cmd.log_dir}",
                title="[bold]STREAM · LM[/bold]",
                border_style="cyan",
                expand=False,
            )
        )

        run_start = time.perf_counter()
        win_start, win_step = run_start, start  # throughput/ETA window
        win_loss = torch.zeros((), device=device)  # accumulate on-device; sync only at log_every
        it = iter(loader)
        step = start  # survives an already-finished budget, where the loop below never runs
        log.info(f"Training loop started — step {start:,} → {steps:,}.")
        with SignalGuard() as guard:
            for step in range(start + 1, steps + 1):
                shape = self._lr_shape(step, lm)
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * shape
                lr = optimizers[-1].param_groups[0]["lr"]
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                # Micro-batches share one optimiser step. Every micro-batch holds the same token
                # count (fixed context_len, drop_last), so the mean of the scaled per-micro losses
                # is exactly the full-batch loss -- CE and the z-loss term are both per-position
                # means -- and the accumulated gradient is the full-batch gradient, not an
                # approximation of it.
                step_loss = torch.zeros((), device=device)
                for _ in range(lm.grad_accum):
                    try:
                        x, y, seg = next(it)
                    except StopIteration:
                        it = iter(loader)
                        x, y, seg = next(it)
                    x, y, seg = x.to(device), y.to(device), seg.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                        logits = model(x, segments=seg)
                        flat = logits.reshape(-1, logits.shape[-1])
                        loss = torch.nn.functional.cross_entropy(flat, y.reshape(-1))
                        if lm.z_loss:
                            # log Z is exactly the log-sum-exp the cross-entropy already computes;
                            # squaring it pulls the softmax normaliser back toward 1 and keeps the
                            # logits from drifting.
                            loss = loss + lm.z_loss * flat.logsumexp(dim=-1).pow(2).mean()
                        loss = loss / lm.grad_accum
                    loss.backward()
                    step_loss += loss.detach()  # kept on-device; no per-micro sync
                torch.nn.utils.clip_grad_norm_(model.parameters(), lm.grad_clip)
                for opt in optimizers:
                    opt.step()
                win_loss += step_loss

                if step % lm.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    avg_loss = (win_loss / max(step - win_step, 1)).item()  # one sync per window
                    eta = _fmt_hms((steps - step) / its) if its > 0 else "—"
                    pct = 100.0 * step / steps
                    log.info(
                        f"step {step:>7,}/{steps:,} ({pct:4.1f}%) │ "
                        f"loss {avg_loss:7.3f} │ lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/it_per_s", its, step)
                    win_start, win_step = now, step
                    win_loss.zero_()

                if step % lm.eval_interval == 0 or step == steps:
                    ppl = self._perplexity(model, val, micro, device)
                    writer.add_scalar("val/ppl", ppl, step)
                    improved = ppl < best
                    best = min(best, ppl)
                    marker = "  ← best" if improved else f"  (best {best:.3f})"
                    log.log(
                        "SUCCESS" if improved else "INFO",
                        f"val ppl {ppl:8.3f}{marker}  @ step {step:,}",
                    )
                    if improved:
                        self._save(best_ckpt, model, optimizers, step, resume_count, ppl)

                if step % lm.ckpt_every == 0:
                    self._save(last_ckpt, model, optimizers, step, resume_count, best)
                if guard.stop_requested:
                    self._save(last_ckpt, model, optimizers, step, resume_count, best)
                    log.warning(f"interrupt received — checkpointed @ step {step:,}; exiting.")
                    writer.close()
                    return best

        self._save(last_ckpt, model, optimizers, step, resume_count, best)
        writer.close()
        console.print(
            Panel(
                f"steps {step:,} · elapsed {_fmt_hms(time.perf_counter() - run_start)} · "
                f"best val ppl {best if math.isfinite(best) else float('nan'):.3f}\n"
                f"last → {last_ckpt}\nbest → {best_ckpt}",
                title="[bold green]STREAM-LM complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        return best

    def _save(
        self,
        path: str,
        model: StreamLmModel,
        optimizers: list[torch.optim.Optimizer],
        step: int,
        resume_count: int,
        val_ppl: float,
    ) -> None:
        save_checkpoint(
            path,
            model,
            optimizers,
            step,
            resume_count=resume_count,
            kind="lm",
            extra={"val_ppl": val_ppl},
        )

    def _loader(
        self, bin_path: str, ctx: int, batch: int, seed: int
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # replacement=True is load-bearing at corpus scale: the whole LibriSpeech-LM corpus packs
        # to ~1.6e9 windows, and shuffle=True (replacement=False) would make RandomSampler
        # materialize torch.randperm(1.6e9).tolist() — a ~13 GB tensor plus a billion-element
        # Python list — which OOM-swaps before step 1 (RAM pinned, GPU idle). Sampling with
        # replacement draws lazy torch.randint chunks instead (flat memory), and at 1.6e9 windows
        # the collision rate over a run is negligible — the nanoGPT-style sampling this file wants.
        ds = LmDataset(bin_path, ctx)
        # Explicit generator rather than the global RNG: resume_if_available restores the RNG state
        # the interrupted run was at, which would make the restart replay the same windows. Seeding
        # from lm.seed + resume_count instead makes the stream reproducible from config alone and
        # gives every resume a different stretch of it.
        return DataLoader(
            ds,
            batch_size=batch,
            sampler=RandomSampler(
                ds, replacement=True, generator=torch.Generator().manual_seed(seed)
            ),
            drop_last=True,
        )

    def _optimizers(self, model: StreamLmModel, lm: LmConfig) -> list[torch.optim.Optimizer]:
        # Muon on the block weight matrices, AdamW on the tied embedding/readout and the norms --
        # the same split the acoustic stack uses, which is why partition_params is shared.
        # `head` is named as a readout so the tied table is stepped once, by AdamW.
        if lm.optimizer == "adamw":
            return [
                torch.optim.AdamW(
                    model.parameters(),
                    lr=lm.lr_peak,
                    weight_decay=lm.weight_decay,
                    betas=(0.9, 0.95),
                )
            ]
        muon_p, adamw_p = partition_params(model, head_patterns=("head",))
        return [
            Muon(muon_p, lr=lm.muon_lr, weight_decay=lm.weight_decay),
            torch.optim.AdamW(
                adamw_p, lr=lm.lr_peak, weight_decay=lm.weight_decay, betas=(0.9, 0.95)
            ),
        ]

    def _lr_shape(self, step: int, lm: LmConfig) -> float:
        # Warmup -> cosine decay as a multiplier in [0, 1] on each group's own peak LR.
        if step < lm.warmup_steps:
            return step / max(1, lm.warmup_steps)
        prog = (step - lm.warmup_steps) / max(1, lm.total_steps - lm.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    @torch.no_grad()
    def _perplexity(self, model: StreamLmModel, val: LmDataset, batch: int, device: str) -> float:
        model.eval()
        loader = DataLoader(val, batch_size=batch, drop_last=True)
        total, count = 0.0, 0
        for x, y, seg in loader:
            # Bounded in WINDOWS, not batches, so val ppl scores the same slice of val.bin
            # regardless of what grad_accum does to the micro-batch size.
            if count * batch >= _VAL_WINDOWS:
                break
            x, y, seg = x.to(device), y.to(device), seg.to(device)
            loss = torch.nn.functional.cross_entropy(
                model(x, segments=seg).reshape(-1, model.vocab), y.reshape(-1)
            )
            total += float(loss)
            count += 1
        model.train()
        return math.exp(total / max(1, count))
