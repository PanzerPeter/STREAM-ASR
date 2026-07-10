# src/slices/TrainLanguageModel/TrainLm_Handler.py
# STREAM-LM training: AdamW + warmup->cosine, bf16 autocast (eager), val perplexity, best-ckpt.
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
        opt = torch.optim.AdamW(
            model.parameters(), lr=lm.lr_peak, weight_decay=lm.weight_decay, betas=(0.9, 0.95)
        )
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
                f"batch {lm.batch_size} · lr {lm.lr_peak:g} / warmup {lm.warmup_steps:,} · "
                f"eval every {lm.eval_interval:,}\n"
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
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            lr = self._lr(step, lm)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), lm.grad_clip)
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
                    save_checkpoint(best_ckpt, model, opt, step, {"val_ppl": ppl})

        save_checkpoint(last_ckpt, model, opt, steps, {"val_ppl": best})
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
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
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

    def _lr(self, step: int, lm: LmConfig) -> float:
        if step < lm.warmup_steps:
            return lm.lr_peak * step / max(1, lm.warmup_steps)
        prog = (step - lm.warmup_steps) / max(1, lm.total_steps - lm.warmup_steps)
        return 0.5 * lm.lr_peak * (1 + math.cos(math.pi * min(1.0, prog)))

    @torch.no_grad()
    def _perplexity(self, model: StreamLmModel, val: LmDataset, lm: LmConfig, device: str) -> float:
        model.eval()
        loader = DataLoader(val, batch_size=lm.batch_size, drop_last=True)
        total, count = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= 20:  # bounded eval subset
                break
            x, y = x.to(device), y.to(device)
            loss = torch.nn.functional.cross_entropy(
                model(x).reshape(-1, model.vocab), y.reshape(-1)
            )
            total += float(loss)
            count += 1
        model.train()
        return math.exp(total / max(1, count))
