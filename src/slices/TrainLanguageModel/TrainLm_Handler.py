# src/slices/TrainLanguageModel/TrainLm_Handler.py
# STREAM-LM training: Muon+AdamW + warmup->cosine, bf16 autocast (eager), z-loss, val perplexity,
# best-ckpt. Windows are document-masked, so a training position only ever attends its own corpus
# line -- the same context a rescored ASR hypothesis has at decode time.
# Terminal logging mirrors Stage A/B (Logging_Adapter loguru sink + rich Panels + TensorBoard)
# so a multi-hour LM run is monitored the same way as the acoustic-model runs.
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.panel import Panel

from src.shared_kernel.Config_Adapter import LmConfig, get_config
from src.shared_kernel.Checkpoint_Adapter import save_checkpoint
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Muon_Optimizer import Muon
from src.shared_kernel.Optimizer_Adapter import partition_params
from src.slices.TrainLanguageModel.LmDataset import LmDataset
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command


def _fmt_hms(seconds: float) -> str:
    # Duplicated from Stage A/B's identical helper rather than imported: AC-002 forbids reaching
    # into another slice's internals, and VSA makes duplication (not a shared-kernel promotion)
    # the default until an explicit /abstract command.
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class TrainLm_Handler:
    def run(self, cmd: TrainLm_Command) -> float:
        log = configure_logging()
        console = Console()
        lm = get_config().lm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        log.info(f"STREAM-LM training on <{device}>")

        model = StreamLmModel().to(device)
        optimizers = self._optimizers(model, lm)
        # Peak LR per group, captured once: the schedule below is a SHAPE in [0, 1] that multiplies
        # them, so Muon's and AdamW's very different peaks keep their ratio through warmup + decay.
        peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
        steps = min(cmd.max_steps, lm.total_steps)
        loader = self._loader(cmd.train_bin, lm.context_len, lm.batch_size)
        val = LmDataset(cmd.val_bin, lm.context_len)
        Path(cmd.out_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(cmd.log_dir)

        n_params = sum(p.numel() for p in model.parameters())
        console.print(
            Panel(
                f"device {device} · params {n_params / 1e6:.1f} M · steps {steps:,}\n"
                f"d_model {lm.d_model} · layers {lm.layers} · heads {lm.heads} "
                f"(kv {lm.kv_groups}) · ctx {lm.context_len}\n"
                f"batch {lm.batch_size} · {lm.optimizer} · adamw lr {lm.lr_peak:g} / muon lr "
                f"{lm.muon_lr:g} / warmup {lm.warmup_steps:,} · eval every {lm.eval_interval:,}\n"
                f"ckpt → {cmd.out_dir} · tb → {cmd.log_dir}",
                title="[bold]STREAM · LM[/bold]",
                border_style="cyan",
                expand=False,
            )
        )

        best = math.inf
        best_ckpt = f"{cmd.out_dir}/lm_best.pt"
        last_ckpt = f"{cmd.out_dir}/lm_last.pt"
        run_start = time.perf_counter()
        win_start, win_step = run_start, 0  # throughput/ETA window
        win_loss = torch.zeros((), device=device)  # accumulate on-device; sync only at log_every
        it = iter(loader)
        log.info(f"Training loop started — target {steps:,} steps.")
        for step in range(1, steps + 1):
            try:
                x, y, seg = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, seg = next(it)
            x, y, seg = x.to(device), y.to(device), seg.to(device)
            shape = self._lr_shape(step, lm)
            for opt, peaks in zip(optimizers, peak_lrs):
                for g, peak in zip(opt.param_groups, peaks):
                    g["lr"] = peak * shape
            lr = optimizers[-1].param_groups[0]["lr"]
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x, segments=seg)
                flat = logits.reshape(-1, logits.shape[-1])
                loss = torch.nn.functional.cross_entropy(flat, y.reshape(-1))
                if lm.z_loss:
                    # log Z is exactly the log-sum-exp the cross-entropy already computes; squaring
                    # it pulls the softmax normaliser back toward 1 and keeps logits from drifting.
                    loss = loss + lm.z_loss * flat.logsumexp(dim=-1).pow(2).mean()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), lm.grad_clip)
            for opt in optimizers:
                opt.step()
            win_loss += loss.detach()  # kept on-device; no per-step sync

            if step % lm.log_every == 0:
                now = time.perf_counter()
                its = (step - win_step) / max(now - win_start, 1e-9)
                avg_loss = (win_loss / max(step - win_step, 1)).item()  # single sync per window
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
                ppl = self._perplexity(model, val, lm, device)
                writer.add_scalar("val/ppl", ppl, step)
                improved = ppl < best
                best = min(best, ppl)
                marker = "  ← best" if improved else f"  (best {best:.3f})"
                log.log(
                    "SUCCESS" if improved else "INFO",
                    f"val ppl {ppl:8.3f}{marker}  @ step {step:,}",
                )
                if improved:
                    save_checkpoint(best_ckpt, model, optimizers, step, extra={"val_ppl": ppl})

        save_checkpoint(last_ckpt, model, optimizers, steps, extra={"val_ppl": best})
        writer.close()
        console.print(
            Panel(
                f"steps {steps:,} · elapsed {_fmt_hms(time.perf_counter() - run_start)} · "
                f"best val ppl {best if math.isfinite(best) else float('nan'):.3f}\n"
                f"last → {last_ckpt}\nbest → {best_ckpt}",
                title="[bold green]STREAM-LM complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        return best

    def _loader(
        self, bin_path: str, ctx: int, batch: int
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # replacement=True is load-bearing at corpus scale: the whole LibriSpeech-LM corpus packs
        # to ~1.6e9 windows, and shuffle=True (replacement=False) would make RandomSampler
        # materialize torch.randperm(1.6e9).tolist() — a ~13 GB tensor plus a billion-element
        # Python list — which OOM-swaps before step 1 (RAM pinned, GPU idle). Sampling with
        # replacement draws lazy torch.randint chunks instead (flat memory), and at 1.6e9 windows
        # the collision rate over a run is negligible — the nanoGPT-style sampling this file wants.
        ds = LmDataset(bin_path, ctx)
        return DataLoader(
            ds,
            batch_size=batch,
            sampler=RandomSampler(ds, replacement=True),
            drop_last=True,
        )

    def _optimizers(self, model: StreamLmModel, lm: LmConfig) -> list[torch.optim.Optimizer]:
        # Muon on the block weight matrices, AdamW on the tied embedding/readout and the norms --
        # the same split the acoustic stack uses (SP3), which is why partition_params is shared.
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
    def _perplexity(self, model: StreamLmModel, val: LmDataset, lm: LmConfig, device: str) -> float:
        model.eval()
        loader = DataLoader(val, batch_size=lm.batch_size, drop_last=True)
        total, count = 0.0, 0
        for i, (x, y, seg) in enumerate(loader):
            if i >= 20:  # bounded eval subset
                break
            x, y, seg = x.to(device), y.to(device), seg.to(device)
            loss = torch.nn.functional.cross_entropy(
                model(x, segments=seg).reshape(-1, model.vocab), y.reshape(-1)
            )
            total += float(loss)
            count += 1
        model.train()
        return math.exp(total / max(1, count))
