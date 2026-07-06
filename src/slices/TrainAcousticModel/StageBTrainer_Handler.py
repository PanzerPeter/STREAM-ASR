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

from src.shared_kernel.Checkpoint_Adapter import save_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.TrainAcousticModel.StageATrainer_Handler import _lr_at, _fmt_hms, _Checkpointed
from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand


def _warm_start(model: HybridCtcAttention, path: str, log) -> None:
    # Load the Stage-A encoder + CTC head; leave the fresh decoder untouched. Stage-A saved a full
    # AcousticModel state_dict ("encoder.*", "ctc_head.*"); those keys map 1:1 onto the hybrid.
    if not path or not os.path.isfile(path):
        log.warning(f"warm_start '{path}' absent — training encoder from scratch.")
        return
    state = torch.load(path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    enc = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
    ctc = {k[len("ctc_head.") :]: v for k, v in sd.items() if k.startswith("ctc_head.")}
    model.encoder.load_state_dict(enc, strict=True)
    model.ctc_head.load_state_dict(ctc, strict=True)
    log.info(f"warm-started encoder + CTC head from {path}")


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

    train_ds = LibriSpeechDataset(cmd.train_manifest, tokenizer, train=True)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=FrameBucketSampler(cmd.train_manifest, sb.max_frames_per_batch, shuffle=True),
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=sb.lr_peak,
        weight_decay=sb.weight_decay,
        betas=(0.9, 0.98),
        fused=(device == "cuda"),
    )

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
    best_wer, best_attn, step = math.inf, math.inf, 0
    run_start = time.perf_counter()
    win_start, win_step = run_start, 0
    model.train()
    while step < cmd.total_steps:
        for batch in train_loader:
            lr = _lr_at(step, sb.lr_peak, sb.warmup_steps, cmd.total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            chunk = random.choice(sb.chunk_sizes)  # dynamic-chunk regularization per batch

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                total, ctc, attn = model.joint_loss(batch, chunk)
                loss = total / sb.grad_accum
            loss.backward()

            if (step + 1) % sb.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), sb.grad_clip)
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
                # Two independent gates: greedy-CTC WER guards the first pass / encoder; attention
                # CE guards the rescorer (Stage B's real deliverable, which greedy-CTC can't see).
                # Save both bests so Phase-2 two-pass eval has the right candidate to pick from.
                if best:
                    save_checkpoint(best_ckpt, model, opt, step)
                if best_a:
                    save_checkpoint(best_attn_ckpt, model, opt, step)
                log.log(
                    "SUCCESS" if (best or best_a) else "INFO",
                    f"dev WER {wer:.4f}{'  ← best' if best else f'  (best {best_wer:.4f})'} │ "
                    f"attn_ce {attn_ce:.4f}"
                    f"{'  ← best' if best_a else f'  (best {best_attn:.4f})'} │ "
                    f"blank {blank_frac:.3f} @ step {step:,}",
                )
                if step >= sb.escape_check_step and blank_frac > sb.escape_max_blank_frac:
                    writer.close()
                    raise RuntimeError(
                        f"CTC blank collapse: blank_frac {blank_frac:.3f} at step {step:,}."
                    )
            if step > 0 and step % sb.ckpt_every == 0:
                save_checkpoint(last_ckpt, model, opt, step)

            step += 1
            if step >= cmd.total_steps:
                break

    save_checkpoint(last_ckpt, model, opt, step)
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
