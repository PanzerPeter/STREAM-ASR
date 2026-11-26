# STREAM-LM training: Muon + AdamW under a WSD schedule with kept branch points, bf16 autocast with
# per-block compile on CUDA, z-loss, whole-val-set perplexity, best-ckpt. Windows are
# document-masked, so a training position only ever attends [EOS] + its own corpus line -- the same
# context a rescored ASR hypothesis has at decode time.
# Terminal logging mirrors the acoustic trainers (Logging_Adapter loguru sink + TensorBoard) so a
# multi-hour LM run is monitored the same way.
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint
from src.shared_kernel.Config_Adapter import LmConfig, get_config
from src.shared_kernel.GradientClipping import (
    clip_grads_per_tensor,
    guarded_parameters,
    unguarded_parameters,
)
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.LrSchedule import lr_at
from src.shared_kernel.Muon_Optimizer import Muon
from src.shared_kernel.Optimizer_Adapter import lr_groups, partition_params
from src.shared_kernel.SignalGuard import SignalGuard
from src.slices.TrainLanguageModel.LmDataset import LmDataset
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.TrainLanguageModel.TrainLm_Command import TrainLm_Command

# dynamo's recompile limit is per code object, and one _Block.forward serves all 16 blocks: with and
# without the value residual, times grad modes. Exceeding it does not raise, it silently demotes the
# block to eager for the rest of the process.
_RECOMPILE_LIMIT = 64


class TrainLm_Handler:
    def run(self, cmd: TrainLm_Command) -> float:
        # Same allocator setting the transducer trainer uses: micro-batch activation blocks vary in
        # size across the run, and growing a segment in place beats rounding up to a fresh block.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log = configure_logging()
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
        # Matrices clip as one group; vectors and scalars per tensor, so no single scalar's gradient
        # sets the factor every matrix (and Muon's momentum EMA) is scaled by.
        matrices, vectors = guarded_parameters(model), unguarded_parameters(model)
        steps = cmd.total_steps
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
        # Compile only on CUDA: it buys GPU kernel fusion, and on CPU (tests) it only costs time.
        # nn.Module.compile keeps state_dict keys, so checkpoints load into the eager decode path.
        n_compiled = self._compile_blocks(model) if lm.compile_blocks and device == "cuda" else 0
        loader = self._loader(cmd.train_bin, lm.context_len, micro, lm.seed + resume_count)
        val = LmDataset(cmd.val_bin, lm.context_len)
        writer = SummaryWriter(cmd.log_dir)

        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            f"STREAM · LM │ device {device} · params {n_params / 1e6:.1f} M · "
            f"vocab {lm.vocab_size} · steps {start:,} → {steps:,}"
        )
        log.info(
            f"d_model {lm.d_model} · layers {lm.layers} · heads {lm.heads} "
            f"(kv {lm.kv_groups}) · ctx {lm.context_len} · compiled blocks {n_compiled or 'off'}"
        )
        log.info(
            f"batch {lm.batch_size} ({lm.grad_accum} x {micro}) · adamw lr {lm.lr_peak:g} / "
            f"muon lr {lm.muon_lr:g} / warmup {lm.warmup_steps:,} / WSD decay last "
            f"{lm.lr_decay_frac:.0%} · branch points {list(lm.branch_points) or 'none'}"
        )
        log.info(
            f"train {len(loader):,} micro-batches per pass · "
            f"val windows {len(val):,} · eval every {lm.eval_interval:,}"
        )
        log.info(
            f"resume {'#' + str(resume_count) if start else 'fresh'} · "
            f"ckpt every {lm.ckpt_every:,} → {cmd.out_dir} · tb → {cmd.log_dir}"
        )

        run_start = time.perf_counter()
        win_start, win_step = run_start, start  # throughput/ETA window
        win_loss = torch.zeros((), device=device)  # accumulate on-device; sync only at log_every
        grad_norm = torch.zeros((), device=device)
        it = iter(loader)
        step = start  # survives an already-finished budget, where the loop below never runs
        log.info(f"Training loop started: step {start:,} -> {steps:,}.")
        with SignalGuard() as guard:
            for step in range(start + 1, steps + 1):
                shape = lr_at(step, 1.0, lm.warmup_steps, steps, decay_frac=lm.lr_decay_frac)
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * shape
                lr = optimizers[-1].param_groups[0]["lr"]  # AdamW
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                # Micro-batches share one optimiser step. Every micro-batch holds the same token
                # count (fixed context_len, drop_last), so the mean of the scaled per-micro losses
                # is exactly the full-batch loss -- CE and the z-loss term are both per-position
                # means -- and the accumulated gradient is the full-batch gradient, not an
                # approximation of it.
                step_ce = torch.zeros((), device=device)
                for _ in range(lm.grad_accum):
                    try:
                        x, y, seg = next(it)
                    except StopIteration:  # a full pass done; the next one is a fresh shuffle
                        it = iter(loader)
                        x, y, seg = next(it)
                    x, y, seg = x.to(device), y.to(device), seg.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                        logits = model(x, segments=seg)
                    ce, loss = self._loss(logits, y, lm.z_loss)
                    (loss / lm.grad_accum).backward()
                    step_ce += ce.detach() / lm.grad_accum  # kept on-device; no per-micro sync
                grad_norm = torch.nn.utils.clip_grad_norm_(matrices, lm.grad_clip)
                clip_grads_per_tensor(vectors, lm.grad_clip)
                for opt in optimizers:
                    opt.step()
                win_loss += step_ce

                if step % lm.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    avg_loss = (win_loss / max(step - win_step, 1)).item()  # one sync per window
                    eta = str(timedelta(seconds=int((steps - step) / its))) if its > 0 else "-"
                    pct = 100.0 * step / steps
                    log.info(
                        f"step {step:>7,}/{steps:,} ({pct:4.1f}%) │ "
                        f"ce {avg_loss:7.3f} │ lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), step)
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
                if step in lm.branch_points:
                    # The trunk at a shorter budget's decay start, never overwritten: copy it to a
                    # fresh out-dir as lm_last.pt and resume with that budget to anneal it.
                    branch = f"{cmd.out_dir}/lm_branch{step}.pt"
                    self._save(branch, model, optimizers, step, resume_count, best)
                    log.info(f"branch point kept → {branch}")
                if guard.stop_requested:
                    self._save(last_ckpt, model, optimizers, step, resume_count, best)
                    log.warning(f"interrupt received; checkpointed @ step {step:,}, exiting.")
                    writer.close()
                    return best

        self._save(last_ckpt, model, optimizers, step, resume_count, best)
        writer.close()
        elapsed = timedelta(seconds=int(time.perf_counter() - run_start))
        log.success(
            f"STREAM-LM complete │ steps {step:,} · elapsed {elapsed} · "
            f"best val ppl {best if math.isfinite(best) else float('nan'):.3f}"
        )
        log.success(f"last → {last_ckpt} · best → {best_ckpt}")
        return best

    @staticmethod
    def _loss(
        logits: torch.Tensor, y: torch.Tensor, z_loss: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In fp32, with ONE log-normaliser serving both terms: CE = logZ - logit[y], and the z-loss
        # (Chowdhery et al.) penalises logZ^2 to hold the partition function near 1.
        flat = logits.float().reshape(-1, logits.shape[-1])
        log_z = flat.logsumexp(dim=-1)
        ce = (log_z - flat.gather(-1, y.reshape(-1, 1)).squeeze(-1)).mean()
        return ce, (ce + z_loss * log_z.pow(2).mean()) if z_loss else ce

    @staticmethod
    def _compile_blocks(model: StreamLmModel) -> int:
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, _RECOMPILE_LIMIT
        )
        for block in model.blocks:
            # Static shapes: every training micro-batch is [micro, context_len].
            block.compile(dynamic=False)
        return len(model.blocks)

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
        # Non-overlapping windows shuffled WITHOUT replacement: one pass reads every token exactly
        # once. (Random offsets drawn with replacement leave e^-1 = 37 % of the tokens unread over a
        # one-pass budget.) The permutation is over ~9e6 windows, not one per token, so it is a
        # 72 MB tensor rather than the 13 GB that a per-offset randperm would need.
        ds = LmDataset(bin_path, ctx)
        # Explicit generator rather than the global RNG: resume_if_available restores the RNG state
        # the interrupted run was at, which would make the restart replay the same windows. Seeding
        # from lm.seed + resume_count instead makes the order reproducible from config alone and
        # gives every resume a fresh permutation.
        return DataLoader(
            ds,
            batch_size=batch,
            sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(seed)),
            drop_last=True,
        )

    def _optimizers(self, model: StreamLmModel, lm: LmConfig) -> list[torch.optim.Optimizer]:
        # Muon on the block weight matrices; AdamW on the embedding, the readout (`head`, named so
        # partition_params keeps it off Muon) and the value-residual scalars. lr_groups exempts
        # every vector and scalar from weight decay: 0 is no neutral target for a mixing weight.
        muon_p, adamw_p = partition_params(model, head_patterns=("head",))
        fused = next(model.parameters()).is_cuda  # fused AdamW kernel on CUDA; False on CPU (tests)
        return [
            Muon(muon_p, lr=lm.muon_lr, weight_decay=lm.weight_decay),
            torch.optim.AdamW(
                lr_groups(adamw_p, lm.lr_peak, lm.weight_decay),
                lr=lm.lr_peak,
                betas=(0.9, 0.95),
                fused=fused,
            ),
        ]

    @torch.no_grad()
    def _perplexity(self, model: StreamLmModel, val: LmDataset, batch: int, device: str) -> float:
        # Every window of val.bin, token-weighted, eager. Each window is a distinct stretch of the
        # bin, so the whole val set is scored once (the first val_words of the interleaved corpus,
        # i.e. a proportional sample of every source).
        model.eval()
        total = torch.zeros((), device=device)
        count = 0
        with torch.compiler.set_stance("force_eager"):
            for x, y, seg in DataLoader(val, batch_size=batch):
                x, y, seg = x.to(device), y.to(device), seg.to(device)
                logits = model(x, segments=seg).float()
                total += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
                )
                count += y.numel()
        model.train()
        return math.exp(total.item() / count)
