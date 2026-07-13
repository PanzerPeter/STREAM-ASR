# src/slices/TrainAcousticModel/StageBTrainer_Handler.py
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

from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.SignalGuard import SignalGuard
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.TrainAcousticModel.StageATrainer_Handler import (
    _lr_at,
    _fmt_hms,
    _Checkpointed,
    _seed_all,
)
from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand


def _assert_phase0_delta(missing: list[str], unexpected: list[str]) -> None:
    # Plan-3b made the encoder causal, which renamed exactly two families of encoder params:
    # the frontend convs (nn.Sequential `conv.0/conv.2` -> causal `conv1/conv2`) and ConvModule's
    # norm (LayerNorm per-channel `weight` -> BiasNorm scalar `log_scale`). A pre-Plan-3b Stage-A
    # checkpoint is therefore expected to miss the new names and carry the old ones; Phase 0 exists
    # to retrain those submodules fresh. Anything missing/unexpected *outside* that delta means the
    # checkpoint no longer matches this encoder -- a real regression that must fail loud rather than
    # silently warm-start a mostly-random encoder.
    bad_missing = [
        k
        for k in missing
        if not (
            k.startswith("frontend.conv1.")
            or k.startswith("frontend.conv2.")
            or k.endswith(".conv.norm.log_scale")
        )
    ]
    bad_unexpected = [
        k
        for k in unexpected
        if not (k.startswith("frontend.conv.") or k.endswith(".conv.norm.weight"))
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "warm-start state_dict mismatch beyond the Plan-3b Phase-0 delta — "
            f"unexpected missing keys: {bad_missing}; unexpected extra keys: {bad_unexpected}"
        )


def _warm_start(model: HybridCtcAttention, path: str, log) -> None:
    # Load the Stage-A encoder + CTC head; leave the fresh decoder untouched. Stage-A saved a full
    # AcousticModel state_dict ("encoder.*", "ctc_head.*"). Most keys map 1:1 onto the hybrid, but
    # the Plan-3b causal rework renamed the frontend convs and swapped ConvModule's norm, so the
    # encoder loads non-strict and re-inits that known delta (validated by _assert_phase0_delta).
    if not path or not os.path.isfile(path):
        log.warning(f"warm_start '{path}' absent — training encoder from scratch.")
        return
    state = torch.load(path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    enc = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
    ctc = {k[len("ctc_head.") :]: v for k, v in sd.items() if k.startswith("ctc_head.")}
    result = model.encoder.load_state_dict(enc, strict=False)
    _assert_phase0_delta(list(result.missing_keys), list(result.unexpected_keys))
    model.ctc_head.load_state_dict(ctc, strict=True)  # CTC head is unchanged by Plan 3b -> strict
    log.info(
        f"warm-started encoder + CTC head from {path} "
        f"({len(result.missing_keys)} Phase-0 encoder params re-init fresh)"
    )


@torch.no_grad()
def _dev_wer(model, loader, tokenizer, device) -> tuple[float, float, float]:
    """Greedy-CTC WER + blank fraction (first-pass / encoder health) and teacher-forced attention
    cross-entropy (decoder health). The CE is the cheap proxy for the two-pass rescoring quality
    that greedy-CTC WER is blind to — the decoder can improve rescoring while CTC WER plateaus, and
    it can overfit while CTC WER holds. Token-weighted mean over the dev set."""
    blank_id = get_config().model.blank_id
    model.eval()
    refs, hyps, blank_frames, total_frames = [], [], 0, 0
    attn_ce_sum, attn_tok = 0.0, 0
    for batch in loader:
        ctc_logits, memory, out_len = model(
            batch.features.to(device), batch.feature_lengths.to(device)
        )
        best = ctc_logits.float().cpu().argmax(dim=-1)
        for b in range(best.shape[0]):
            valid = int(out_len[b])
            blank_frames += int((best[b, :valid] == blank_id).sum())
            total_frames += valid
        hyps.extend(ctc_greedy_decode(ctc_logits.float().cpu(), out_len.cpu(), tokenizer))
        for i in range(batch.tokens.shape[0]):
            refs.append(tokenizer.decode(batch.tokens[i, : batch.token_lengths[i]].tolist()))

        n_tok = int(batch.dec_lengths.sum())  # non-ignored CE positions in this batch
        attn = model.attention_loss(
            memory,
            out_len,
            FeatureBatch(
                batch.features,
                batch.feature_lengths,
                batch.tokens,
                batch.token_lengths,
                batch.dec_in_l2r.to(device),
                batch.dec_out_l2r.to(device),
                batch.dec_in_r2l.to(device),
                batch.dec_out_r2l.to(device),
                batch.dec_lengths.to(device),
            ),
        )
        attn_ce_sum += attn.item() * n_tok
        attn_tok += n_tok
    model.train()
    return (
        jiwer.wer(refs, hyps),
        blank_frames / max(total_frames, 1),
        attn_ce_sum / max(attn_tok, 1),
    )


def run_stage_b(cmd: StageBTrainCommand) -> str:
    log = configure_logging()
    console = Console()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    writer = SummaryWriter(cmd.log_dir)
    sb = get_config().training.stage_b
    _seed_all(sb.seed)
    log.info(f"seed {sb.seed} (reproducible init/augment/batch order)")

    train_ds = LibriSpeechDataset(cmd.train_manifest, tokenizer, train=True)
    # Named (not inline) so it can be reseeded post-construction after a resume (see below) --
    # FrameBucketSampler reads self._seed lazily in __iter__, so this takes effect pre-loop.
    train_sampler = FrameBucketSampler(
        cmd.train_manifest, sb.max_frames_per_batch, shuffle=True, seed=sb.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_features,
        num_workers=4,
        pin_memory=True,
    )
    dev_ds = LibriSpeechDataset(cmd.dev_manifest, tokenizer, train=False)
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=FrameBucketSampler(cmd.dev_manifest, sb.max_frames_per_batch),
        collate_fn=collate_features,
    )

    cmvn = cmd.cmvn_path if os.path.isfile(cmd.cmvn_path) else None
    model = HybridCtcAttention(cmvn_path=cmvn).to(device)
    _warm_start(model, cmd.warm_start, log)

    # Optional activation checkpointing to bound VRAM (decoder adds memory pressure vs Stage A).
    # Must wrap AFTER warm-start: wrapping renames stack keys to stacks.N.module.*, which would
    # break the strict encoder state_dict load.
    if sb.grad_checkpoint:
        model.encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in model.encoder.stacks])

    optimizers = build_optimizer(model, get_config().optim)
    # build_optimizer sets each group's lr to its calibrated PEAK (Muon >> AdamW per
    # config/optim.yaml, incl. any muP per-group scaling). Snapshot the peaks so the warmup+cosine
    # schedule is applied as a 0->1->0 SHAPE multiplier per group. A single absolute overwrite would
    # clobber Muon's much larger base LR and any muP ratios, defeating SP3. Optimizer peak LRs come
    # solely from optim.yaml (adamw_lr / muon_lr) — there is no separate per-stage lr_peak.
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    n_params = sum(p.numel() for p in model.parameters())
    console.print(
        Panel(
            f"device {device} · params {n_params / 1e6:.1f} M · steps {cmd.total_steps:,}\n"
            f"chunks {sb.chunk_sizes} · ctc_w {sb.ctc_weight} · rev_w {sb.reverse_weight}\n"
            f"warm-start {cmd.warm_start or '(none)'} · tb → {cmd.log_dir}",
            title="[bold]STREAM · Stage-B[/bold]",
            border_style="cyan",
            expand=False,
        )
    )

    last_ckpt = os.path.join(cmd.ckpt_dir, "stage_b_last.pt")
    best_ckpt = os.path.join(cmd.ckpt_dir, "stage_b_best.pt")
    best_attn_ckpt = os.path.join(cmd.ckpt_dir, "stage_b_best_attn.pt")
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step = int(resumed["step"])
    best_wer = resumed["best_wer"]
    resume_count = int(resumed["resume_count"])
    # best_attn is a secondary gate (attention-CE health) not persisted in checkpoint meta, so it
    # legitimately restarts fresh on resume rather than being restored like best_wer.
    best_attn = math.inf
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
    # Reseed the sampler for a fresh, non-repeating epoch after a resume (read at iteration time
    # by FrameBucketSampler.__iter__, so setting it here before the loop starts takes effect).
    train_sampler._seed = sb.seed + resume_count
    run_start = time.perf_counter()
    win_start, win_step = run_start, 0
    model.train()
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for batch in train_loader:
                lr_shape = _lr_at(step, 1.0, sb.warmup_steps, cmd.total_steps)  # 0->1->0 shape
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                # Representative LR for logging/tensorboard: build_optimizer returns AdamW last
                # ([muon, adamw] or [adamw]), so optimizers[-1] is always the AdamW.
                lr = optimizers[-1].param_groups[0]["lr"]
                chunk = random.choice(sb.chunk_sizes)  # dynamic-chunk regularization per batch

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    total, ctc, attn = model.joint_loss(batch, chunk)
                    loss = total / sb.grad_accum
                loss.backward()

                if (step + 1) % sb.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), sb.grad_clip)
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                if step % sb.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    eta = _fmt_hms((cmd.total_steps - step) / its) if its > 0 else "—"
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} │ loss {total.item():6.3f} "
                        f"(ctc {ctc.item():5.2f} attn {attn.item():5.2f}) │ chunk {chunk} │ "
                        f"lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    writer.add_scalar("train/loss", total.item(), step)
                    writer.add_scalar("train/ctc", ctc.item(), step)
                    writer.add_scalar("train/attn", attn.item(), step)
                    writer.add_scalar("train/lr", lr, step)
                    win_start, win_step = now, step
                if step > 0 and step % sb.val_every == 0:
                    wer, blank_frac, attn_ce = _dev_wer(model, dev_loader, tokenizer, device)
                    writer.add_scalar("dev/wer", wer, step)
                    writer.add_scalar("dev/blank_frac", blank_frac, step)
                    writer.add_scalar("dev/attn_ce", attn_ce, step)
                    best = wer < best_wer
                    best_wer = min(best_wer, wer)
                    best_a = attn_ce < best_attn
                    best_attn = min(best_attn, attn_ce)
                    # Two independent gates: greedy-CTC WER guards the first pass / encoder;
                    # attention CE guards the rescorer (Stage B's real deliverable, which
                    # greedy-CTC can't see). Save both bests so Phase-2 two-pass eval has the
                    # right candidate to pick from.
                    if best:
                        save_checkpoint(
                            best_ckpt,
                            model,
                            optimizers,
                            step,
                            best_wer=best_wer,
                            resume_count=resume_count,
                            kind="stage_b",
                        )
                    if best_a:
                        save_checkpoint(
                            best_attn_ckpt,
                            model,
                            optimizers,
                            step,
                            best_wer=best_wer,
                            resume_count=resume_count,
                            kind="stage_b",
                        )
                    log.log(
                        "SUCCESS" if (best or best_a) else "INFO",
                        f"dev WER {wer:.4f}"
                        f"{'  ← best' if best else f'  (best {best_wer:.4f})'} │ "
                        f"attn_ce {attn_ce:.4f}"
                        f"{'  ← best' if best_a else f'  (best {best_attn:.4f})'} │ "
                        f"blank {blank_frac:.3f} @ step {step:,}",
                    )
                    # Abort only when both signals agree the run is dead (see Stage-A for
                    # rationale): blank still collapsed AND best dev WER never fell below
                    # escape_min_wer_progress.
                    collapsed = blank_frac > sb.escape_max_blank_frac
                    no_progress = best_wer >= sb.escape_min_wer_progress
                    if step >= sb.escape_check_step and collapsed and no_progress:
                        writer.close()
                        raise RuntimeError(
                            f"CTC blank collapse: blank_frac {blank_frac:.3f} and best dev WER "
                            f"{best_wer:.3f} at step {step:,}."
                        )
                if step > 0 and step % sb.ckpt_every == 0:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        best_wer=best_wer,
                        resume_count=resume_count,
                        kind="stage_b",
                    )

                step += 1
                if guard.stop_requested:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        best_wer=best_wer,
                        resume_count=resume_count,
                        kind="stage_b",
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
        kind="stage_b",
    )
    writer.close()
    console.print(
        Panel(
            f"steps {step:,} · elapsed {_fmt_hms(time.perf_counter() - run_start)} · "
            f"best dev WER {best_wer if math.isfinite(best_wer) else float('nan'):.4f} · "
            f"best attn_ce {best_attn if math.isfinite(best_attn) else float('nan'):.4f}\n"
            f"last → {last_ckpt}\nbest(WER) → {best_ckpt}\nbest(attn) → {best_attn_ckpt}",
            title="[bold green]Stage-B complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    return last_ckpt
