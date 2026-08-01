# The joint transducer training stage: warm-start the encoder from BEST-RQ, then train encoder +
# predictor + joiner + CTC/InterCTC heads together under one objective. Runs on the resumable,
# SIGINT-safe harness (Checkpoint_Adapter + SignalGuard) and selects checkpoints on full-dev
# greedy-CTC WER.
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
    # BEST-RQ saves exactly encoder.* weights, and the transducer's encoder is byte-identical to
    # the one it pretrained, so a strict load must match. Predictor/joiner/heads start fresh.
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
                # `new_state` returned here is already the context after consuming `prev`
                # (step's new_state == [state, prev][1:]) -- exactly what the NEXT step needs.
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


def _probe_batches(n_batches: int, n_utts: int, wer_utts: int) -> set[int]:
    # Which dev batches the greedy-transducer probe decodes. `wer_utts` is a budget in utterances;
    # at the loader's mean batch size that is `k` batches, placed evenly over [0, n_batches - 1] so
    # the probe always includes both ends of the duration-sorted range.
    k = max(1, min(n_batches, round(wer_utts * n_batches / max(1, n_utts))))
    return {round(i * (n_batches - 1) / max(1, k - 1)) for i in range(k)}


@torch.no_grad()
def _dev_metrics(
    model: TransducerModel, loader, tokenizer, device: str, wer_utts: int
) -> tuple[float, float, float]:
    # Greedy-CTC WER + blank fraction over the WHOLE dev set (~54k words) plus a greedy-transducer
    # WER over a `wer_utts` subsample (~1k words, so +-0.008 at 1 sigma -- a probe, not a criterion;
    # checkpoint selection uses the ctc_wer above it, see run_transducer).
    #
    # The subsample spreads evenly over the loader rather than taking its head: FrameBucketSampler
    # emits duration-sorted batches, so the leading `wer_utts` utterances are the shortest clips in
    # dev -- on this manifest that head never exceeds 2.6 s against a 32.6 s corpus maximum, and
    # carries 1.3k words. `k` batches sampled across the full range instead cost the same decode
    # budget but reach the whole duration range and 3.6k words.
    blank_id = get_config().model.blank_id
    model.eval()
    ctc_refs, ctc_hyps, blank_frames, total_frames = [], [], 0, 0
    t_refs, t_hyps = [], []
    probe_batches = _probe_batches(len(loader), len(loader.dataset), wer_utts)
    for batch_idx, batch in enumerate(loader):
        memory, out_len, ctc_logits, _, _ = model(
            batch.features.to(device), batch.feature_lengths.to(device)
        )
        # argmax on-device, transferred once: both the blank count and the greedy decode need only
        # the winning id, so moving [B, T] ids costs ~1/500th of shipping the [B, T, V] logits.
        best = ctc_logits.argmax(dim=-1).cpu()
        lengths = out_len.cpu()
        for b in range(best.shape[0]):
            valid = int(lengths[b])
            blank_frames += int((best[b, :valid] == blank_id).sum())
            total_frames += valid
        ctc_hyps.extend(ctc_greedy_decode(best, lengths, tokenizer))
        probe = batch_idx in probe_batches
        for i in range(batch.tokens.shape[0]):
            ref = tokenizer.decode(batch.tokens[i, : batch.token_lengths[i]].tolist())
            ctc_refs.append(ref)
            if probe:
                t_refs.append(ref)
        if probe:
            t_hyps.extend(greedy_transducer_decode(model, memory, out_len, tokenizer))
    model.train()
    ctc_wer = jiwer.wer(ctc_refs, ctc_hyps)
    t_wer = jiwer.wer(t_refs[: len(t_hyps)], t_hyps[: len(t_refs)]) if t_hyps else 1.0
    return t_wer, ctc_wer, blank_frames / max(total_frames, 1)


def run_transducer(cmd: TransducerTrainCommand) -> str:
    # The unpruned RNN-T lattice [B, T, U+1, V] is a large, size-varying allocation; on a 12 GB
    # card the default caching allocator fragments (big reserved-but-unallocated pool, little free)
    # and a denser bucket eventually fails a multi-hundred-MB alloc. expandable_segments lets the
    # allocator grow a segment in place instead of rounding up to a fresh block, removing that
    # cliff. It is read at the first CUDA allocation below, so setdefault here (before .to(device))
    # is in time and still lets an explicit env override win. Allocator-only -- numerics untouched.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = configure_logging()
    console = Console()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    writer = SummaryWriter(cmd.log_dir)
    tr = get_config().training.transducer
    _seed_all(tr.seed)

    train_ds = LibriSpeechDataset(cmd.train_manifest, tokenizer)
    train_sampler = FrameBucketSampler(
        cmd.train_manifest,
        tr.max_frames_per_batch,
        shuffle=True,
        seed=tr.seed,
        max_tokens_per_batch=tr.max_tokens_per_batch,
        max_lattice_per_batch=tr.max_lattice_per_batch,
        token_sort_window=tr.token_sort_window,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_features,
        num_workers=4,
        pin_memory=True,
    )
    dev_ds = LibriSpeechDataset(cmd.dev_manifest, tokenizer)
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=FrameBucketSampler(
            cmd.dev_manifest,
            tr.max_frames_per_batch,
            max_tokens_per_batch=tr.max_tokens_per_batch,
            max_lattice_per_batch=tr.max_lattice_per_batch,
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
            f"lr {tr.lr_schedule} · warmup {tr.warmup_steps:,} · stable {tr.lr_stable_ratio:.2f}"
            f" · decay {tr.lr_decay_frac:.2f} · floor {tr.lr_min_ratio:.3f}\n"
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
    # best_wer is the SELECTION metric (full-dev greedy-CTC WER). Checkpoints written before that
    # switch stored the transducer probe under the same key; the two live in the same numeric range,
    # so resuming one only makes the first few `best` decisions over-strict, never wrong-model.
    best_wer = float(resumed["best_wer"])
    resume_count = int(resumed["resume_count"])
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
    train_sampler._seed = tr.seed + resume_count
    run_start = time.perf_counter()
    win_start, win_step = run_start, 0
    # Kept on-device and only .item()'d at log_every: this scalar is read once per log window, so
    # casting it per optimizer step would buy a device->host sync for nothing.
    last_grad_norm = torch.zeros((), device=device)
    oom_skips = 0
    # Micro-batches accumulated into the current window. Tracked explicitly rather than inferred
    # from `step % grad_accum`, because an OOM drops a partial window: the counter resets with it
    # so the next optimizer step still fires on a full grad_accum worth of (1/grad_accum)-scaled
    # gradient, instead of applying a window that is short by however many batches were discarded.
    accumulated = 0
    model.train()
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for batch in train_loader:
                lr_shape = _lr_at(
                    step,
                    1.0,
                    tr.warmup_steps,
                    cmd.total_steps,
                    schedule=tr.lr_schedule,
                    stable_ratio=tr.lr_stable_ratio,
                    decay_frac=tr.lr_decay_frac,
                    min_ratio=tr.lr_min_ratio,
                )
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                lr = optimizers[-1].param_groups[0]["lr"]
                chunk = random.choice(tr.chunk_sizes)

                try:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                    ):
                        total, rnnt, ctc, ictc, cr = model.joint_loss(batch, chunk)
                        loss = total / tr.grad_accum
                    loss.backward()
                except torch.OutOfMemoryError:
                    # The RNN-T lattice [B,T,U+1,V] is the run's largest allocation, so a dense
                    # frame x token bucket is what fails first when another process takes part of
                    # the card mid-run. That is a transient property of one bucket, not a trend:
                    # dropping it costs one batch out of thousands per epoch and keeps a 100k-step
                    # run alive instead of losing hours to a passing memory spike. The partial
                    # backward left gradients on only some params, so the whole accumulation
                    # window is discarded rather than applied lopsided.
                    oom_skips += 1
                    accumulated = 0
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    del batch
                    torch.cuda.empty_cache()
                    log.warning(
                        f"CUDA OOM @ step {step:,} — batch dropped (skips {oom_skips}). "
                        "Persistent skips mean the budgets in config/training.yaml are too high "
                        "for the free VRAM."
                    )
                    continue

                accumulated += 1
                if accumulated == tr.grad_accum:
                    # NB: clip_grad_norm_ bounds the AdamW params but is effectively inert for the
                    # encoder/joiner matrices on Muon -- Muon renormalises each update by its own
                    # gradient norm, so the pre-clip norm is diagnostic, not a safety bound there.
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), tr.grad_clip
                    )
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    accumulated = 0

                if step % tr.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    eta = _fmt_hms((cmd.total_steps - step) / its) if its > 0 else "—"
                    # Watch the pretrained encoder drift: a fast climb here after warmup is the
                    # erosion signature the encoder_lr_scale / per-token loss fix target.
                    # _foreach_norm folds the ~600 per-parameter reductions into one fused
                    # multi-tensor launch; the norm of the per-tensor norms is the global L2 norm.
                    with torch.no_grad():
                        enc_norm = torch.linalg.vector_norm(
                            torch.stack(torch._foreach_norm(list(model.encoder.parameters())))
                        )
                        # Every .item() drains the CUDA queue, so pulling seven scalars one at a
                        # time costs seven pipeline stalls per log window. Stack them and sync once.
                        # .float() per element, not on the stack: under bf16 autocast the loss
                        # terms are not guaranteed to share a dtype, and torch.stack rejects a
                        # mixed-dtype list outright.
                        scalars = torch.stack(
                            [
                                t.float()
                                for t in (total, rnnt, ctc, ictc, cr, last_grad_norm, enc_norm)
                            ]
                        )
                    v_loss, v_rnnt, v_ctc, v_ictc, v_cr, v_gnorm, v_enc = scalars.tolist()
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} │ loss {v_loss:6.3f} "
                        f"(rnnt {v_rnnt:5.2f} ctc {v_ctc:5.2f} "
                        f"ictc {v_ictc:5.2f} cr {v_cr:5.2f}) │ "
                        f"chunk {chunk} │ lr {lr:.2e} │ {its:5.2f} it/s │ eta {eta}"
                    )
                    for name, val in (
                        ("loss", v_loss),
                        ("rnnt", v_rnnt),
                        ("ctc", v_ctc),
                        (
                            "interctc",
                            v_ictc,
                        ),  # raw mean CTC over the tapped stacks (encoder health)
                        ("cr_ctc", v_cr),  # CR-CTC consistency KL (0 when cr_ctc off)
                        ("lr", lr),
                        ("grad_norm", v_gnorm),
                        ("encoder_param_norm", v_enc),
                    ):
                        writer.add_scalar(f"train/{name}", val, step)
                    win_start, win_step = now, step
                if step > 0 and step % tr.val_every == 0:
                    t_wer, ctc_wer, blank_frac = _dev_metrics(
                        model, dev_loader, tokenizer, device, tr.dev_wer_utts
                    )
                    writer.add_scalar("dev/transducer_wer", t_wer, step)
                    writer.add_scalar("dev/ctc_wer", ctc_wer, step)
                    writer.add_scalar("dev/blank_frac", blank_frac, step)
                    # Select on the FULL-dev CTC WER, not the transducer probe. The probe is ~1k
                    # words (1 sigma ~= 0.008) and `best` is a min over ~90 of them, so it selects
                    # the luckiest noise trough: the 175k-step run picked step 152k, mid-anneal,
                    # whose full-dev ctc_wer was 0.0680 against 0.0610 at 174k -- and that
                    # checkpoint decoded 0.60 WER points worse on test-clean than the tail average.
                    # ctc_wer is ~54k words (1 sigma ~= 0.0011), free (already computed here), and
                    # tracks the transducer head closely enough to rank checkpoints.
                    best = ctc_wer < best_wer
                    best_wer = min(best_wer, ctc_wer)
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
                        f"dev ctc-WER {ctc_wer:.4f}"
                        f"{'  ← best' if best else f'  (best {best_wer:.4f})'} │ "
                        f"transducer-probe {t_wer:.4f} │ blank {blank_frac:.3f} @ step {step:,}",
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
            f"best dev ctc-WER {best_wer:.4f}\nlast → {last_ckpt}\nbest → {best_ckpt}",
            title="[bold green]Transducer training complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    return last_ckpt
