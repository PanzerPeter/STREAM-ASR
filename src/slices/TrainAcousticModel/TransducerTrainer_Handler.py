# src/slices/TrainAcousticModel/TransducerTrainer_Handler.py
import glob
import os
import random
import re
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
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel._train_utils import (
    _lr_at,
    _fmt_hms,
    _Checkpointed,
    _seed_all,
)
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand


def _write_rolling_snapshot(
    ckpt_dir: str,
    model: TransducerModel,
    optimizers,
    step: int,
    best_wer: float,
    resume_count: int,
    keep_last_n: int,
) -> None:
    # Retain the newest `keep_last_n` numbered snapshots (transducer_step{N}.pt) so
    # scripts/average_checkpoints.py can mean the tail of training into one decode checkpoint --
    # the standard ASR "checkpoint averaging" win. Distinct from transducer_last.pt (overwritten
    # every ckpt_every for resume); these are immutable per-step points. keep_last_n <= 0 disables.
    if keep_last_n <= 0:
        return
    snap = os.path.join(ckpt_dir, f"transducer_step{step}.pt")
    save_checkpoint(
        snap,
        model,
        optimizers,
        step,
        best_wer=best_wer,
        resume_count=resume_count,
        kind="transducer",
    )
    existing = glob.glob(os.path.join(ckpt_dir, "transducer_step*.pt"))
    numbered = sorted(
        ((int(m.group(1)), p) for p in existing if (m := re.search(r"step(\d+)\.pt$", p))),
        key=lambda x: x[0],
    )
    for _, path in numbered[:-keep_last_n]:
        os.remove(path)


def _warm_start_encoder(model: TransducerModel, path: str, log) -> None:
    # BEST-RQ (SP4) saves exactly encoder.* weights; the SP5 encoder is byte-identical to the one
    # BEST-RQ pretrained, so a strict load must match. Predictor/joiner/heads start fresh.
    if not path or not os.path.isfile(path):
        log.warning(f"warm_start '{path}' absent — training encoder from scratch.")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model.encoder.load_state_dict(sd, strict=True)
    log.info(f"warm-started encoder from {path}")


@torch.no_grad()
def greedy_transducer_decode(
    model: TransducerModel,
    memory: torch.Tensor,
    out_lengths: torch.Tensor,
    tokenizer,
) -> list[str]:
    # Standard RNN-T greedy: per encoder frame, emit non-blank tokens (advancing the predictor)
    # until blank or max_symbols, then step time. Batched utterance-by-utterance for simplicity.
    blank = get_config().model.blank_id
    max_symbols = get_config().decode.max_symbols
    device = memory.device
    texts = []
    for b in range(memory.shape[0]):
        state = model.predictor.init_state(1, device)
        prev = torch.full((1,), blank, dtype=torch.long, device=device)
        ids: list[int] = []
        for t in range(int(out_lengths[b])):
            enc_t = memory[b, t].unsqueeze(0)  # [1, De]
            emitted = 0
            while emitted < max_symbols:
                # `new_state` returned here is already the context after consuming `prev` (Task 3:
                # step's new_state == [state, prev][1:]) -- i.e. exactly what the NEXT step needs.
                # Reusing it (instead of re-deriving state from the just-emitted `tok`) is both the
                # correctness requirement and avoids a second, redundant predictor forward pass.
                pred_out, new_state = model.predictor.step(state, prev)
                logits = model.joiner.step(enc_t, pred_out)  # [1, V]
                tok = int(logits.argmax(dim=-1))
                if tok == blank:
                    break
                ids.append(tok)
                state = new_state
                prev = torch.full((1,), tok, dtype=torch.long, device=device)
                emitted += 1
        texts.append(tokenizer.decode(ids))
    return texts


@torch.no_grad()
def _dev_metrics(
    model: TransducerModel, loader, tokenizer, device: str, wer_utts: int
) -> tuple[float, float, float]:
    # Cheap: greedy-CTC WER + blank fraction over the whole dev set (encoder health). Real signal:
    # greedy-transducer WER over the first `wer_utts` utterances (the deliverable metric).
    blank_id = get_config().model.blank_id
    model.eval()
    ctc_refs, ctc_hyps, blank_frames, total_frames = [], [], 0, 0
    t_refs, t_hyps, seen = [], [], 0
    for batch in loader:
        memory, out_len, ctc_logits, _, _ = model(
            batch.features.to(device), batch.feature_lengths.to(device)
        )
        best = ctc_logits.float().cpu().argmax(dim=-1)
        for b in range(best.shape[0]):
            valid = int(out_len[b])
            blank_frames += int((best[b, :valid] == blank_id).sum())
            total_frames += valid
        ctc_hyps.extend(ctc_greedy_decode(ctc_logits.float().cpu(), out_len.cpu(), tokenizer))
        for i in range(batch.tokens.shape[0]):
            ref = tokenizer.decode(batch.tokens[i, : batch.token_lengths[i]].tolist())
            ctc_refs.append(ref)
            if seen < wer_utts:
                t_refs.append(ref)
        if seen < wer_utts:
            t_hyps.extend(greedy_transducer_decode(model, memory, out_len, tokenizer))
            seen += batch.tokens.shape[0]
    model.train()
    ctc_wer = jiwer.wer(ctc_refs, ctc_hyps)
    t_wer = jiwer.wer(t_refs[: len(t_hyps)], t_hyps[: len(t_refs)]) if t_hyps else 1.0
    return t_wer, ctc_wer, blank_frames / max(total_frames, 1)


def run_transducer(cmd: TransducerTrainCommand) -> str:
    log = configure_logging()
    console = Console()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    writer = SummaryWriter(cmd.log_dir)
    tr = get_config().training.transducer
    _seed_all(tr.seed)

    train_ds = LibriSpeechDataset(cmd.train_manifest, tokenizer, train=True)
    train_sampler = FrameBucketSampler(
        cmd.train_manifest,
        tr.max_frames_per_batch,
        shuffle=True,
        seed=tr.seed,
        max_tokens_per_batch=tr.max_tokens_per_batch,
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
        batch_sampler=FrameBucketSampler(
            cmd.dev_manifest, tr.max_frames_per_batch, max_tokens_per_batch=tr.max_tokens_per_batch
        ),
        collate_fn=collate_features,
    )

    cmvn = cmd.cmvn_path if os.path.isfile(cmd.cmvn_path) else None
    model = TransducerModel(cmvn_path=cmvn).to(device)
    _warm_start_encoder(model, cmd.warm_start, log)
    if tr.grad_checkpoint:
        model.encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in model.encoder.stacks])

    optimizers = build_optimizer(model, get_config().optim)
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    n_params = sum(p.numel() for p in model.parameters())
    console.print(
        Panel(
            f"device {device} · params {n_params / 1e6:.1f} M · steps {cmd.total_steps:,}\n"
            f"chunks {tr.chunk_sizes} · ctc_aux {get_config().transducer.ctc_aux_weight}\n"
            f"warm-start {cmd.warm_start or '(none)'} · tb → {cmd.log_dir}",
            title="[bold]STREAM · Transducer[/bold]",
            border_style="cyan",
            expand=False,
        )
    )

    last_ckpt = os.path.join(cmd.ckpt_dir, "transducer_last.pt")
    best_ckpt = os.path.join(cmd.ckpt_dir, "transducer_best.pt")
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step = int(resumed["step"])
    best_wer = resumed["best_wer"]
    resume_count = int(resumed["resume_count"])
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
    train_sampler._seed = tr.seed + resume_count
    run_start = time.perf_counter()
    win_start, win_step = run_start, 0
    last_grad_norm = 0.0
    model.train()
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for batch in train_loader:
                lr_shape = _lr_at(step, 1.0, tr.warmup_steps, cmd.total_steps)
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                lr = optimizers[-1].param_groups[0]["lr"]
                chunk = random.choice(tr.chunk_sizes)

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    total, rnnt, ctc, ictc = model.joint_loss(batch, chunk)
                    loss = total / tr.grad_accum
                loss.backward()

                if (step + 1) % tr.grad_accum == 0:
                    # NB: clip_grad_norm_ bounds the AdamW params but is effectively inert for the
                    # encoder/joiner matrices on Muon -- Muon renormalises each update by its own
                    # gradient norm, so the pre-clip norm is diagnostic, not a safety bound there.
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
                    )
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                if step % tr.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    eta = _fmt_hms((cmd.total_steps - step) / its) if its > 0 else "—"
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} │ loss {total.item():6.3f} "
                        f"(rnnt {rnnt.item():5.2f} ctc {ctc.item():5.2f} "
                        f"ictc {ictc.item():5.2f}) │ "
                        f"chunk {chunk} │ lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    for name, val in (
                        ("loss", total),
                        ("rnnt", rnnt),
                        ("ctc", ctc),
                        ("interctc", ictc),  # raw mean CTC over the tapped stacks (encoder health)
                    ):
                        writer.add_scalar(f"train/{name}", val.item(), step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", last_grad_norm, step)
                    # Watch the pretrained encoder drift: a fast climb here after warmup is the
                    # erosion signature the encoder_lr_scale / per-token loss fix target.
                    enc_sq = torch.zeros((), device=device)
                    for p in model.encoder.parameters():
                        enc_sq = enc_sq + p.detach().float().pow(2).sum()
                    writer.add_scalar("train/encoder_param_norm", torch.sqrt(enc_sq).item(), step)
                    win_start, win_step = now, step
                if step > 0 and step % tr.val_every == 0:
                    t_wer, ctc_wer, blank_frac = _dev_metrics(
                        model, dev_loader, tokenizer, device, tr.dev_wer_utts
                    )
                    writer.add_scalar("dev/transducer_wer", t_wer, step)
                    writer.add_scalar("dev/ctc_wer", ctc_wer, step)
                    writer.add_scalar("dev/blank_frac", blank_frac, step)
                    best = t_wer < best_wer
                    best_wer = min(best_wer, t_wer)
                    if best:
                        save_checkpoint(
                            best_ckpt,
                            model,
                            optimizers,
                            step,
                            best_wer=best_wer,
                            resume_count=resume_count,
                            kind="transducer",
                        )
                    log.log(
                        "SUCCESS" if best else "INFO",
                        f"dev transducer-WER {t_wer:.4f}"
                        f"{'  ← best' if best else f'  (best {best_wer:.4f})'} │ "
                        f"ctc-WER {ctc_wer:.4f} │ blank {blank_frac:.3f} @ step {step:,}",
                    )
                if step > 0 and step % tr.ckpt_every == 0:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        best_wer=best_wer,
                        resume_count=resume_count,
                        kind="transducer",
                    )
                    _write_rolling_snapshot(
                        cmd.ckpt_dir,
                        model,
                        optimizers,
                        step,
                        best_wer,
                        resume_count,
                        tr.keep_last_n,
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
                        kind="transducer",
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
        kind="transducer",
    )
    writer.close()
    console.print(
        Panel(
            f"steps {step:,} · elapsed {_fmt_hms(time.perf_counter() - run_start)} · "
            f"best dev transducer-WER {best_wer:.4f}\nlast → {last_ckpt}\nbest → {best_ckpt}",
            title="[bold green]Transducer training complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    return last_ckpt
