# pyright: reportPrivateImportUsage=false
#   torch's `_foreach_*` multi-tensor ops are documented public API with private-looking
#   names -- torch's own optimizers call them -- but they are not in `torch.__all__`, so a
#   strict checker flags every use. mypy accepts them; only this rule needs the exemption.
# The joint transducer training stage: warm-start the encoder from BEST-RQ, then train encoder +
# predictor + joiner + CTC/InterCTC heads together under one objective. Runs on the resumable,
# SIGINT-safe harness (Checkpoint_Adapter + SignalGuard) and selects checkpoints on full-dev
# greedy-CTC WER.
import glob
import math
import os
import random
import re
import shutil
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import jiwer
from rich.console import Console
from rich.panel import Panel

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
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.ParameterProjection import project_constraints
from src.shared_kernel.SignalGuard import SignalGuard
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel._train_utils import (
    _fmt_hms,
    _Checkpointed,
    _seed_all,
    branch_gain_params,
    gains_at_ceiling,
    GainCeilingWatch,
    compile_hot_modules,
    stack_mix_params,
    trunk_gain_max,
    trunk_rms_values,
    trunk_stable_rank_min,
    GradNormGuard,
)
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand


def _open_cache(cache_dir: str, split: str, log) -> FeatureCacheReader | None:
    # Absent cache is survivable -- LibriSpeechDataset recomputes log-mel from FLAC and applies the
    # row's own `speed` -- so warn rather than abort. A cache that IS present but disagrees with the
    # config's front end or its source manifest raises out of here, which is the intended outcome:
    # row order is the cache index, so a mismatched pair trains on other utterances' audio.
    if not split:
        log.info("feature cache disabled; decoding audio per epoch.")
        return None
    if not os.path.isfile(os.path.join(cache_dir, f"{split}.header.json")):
        log.warning(f"no feature cache '{split}' in {cache_dir}; decoding audio per epoch instead.")
        return None
    return FeatureCacheReader(cache_dir, split)


def _write_rolling_snapshot(ckpt_dir: str, source: str, step: int, keep_last_n: int) -> None:
    # Retain the newest `keep_last_n` numbered snapshots (transducer_step{N}.pt) so
    # scripts/average_checkpoints.py can mean the tail of training into one decode checkpoint --
    # the standard ASR "checkpoint averaging" win. Distinct from transducer_last.pt (overwritten
    # every ckpt_every for resume); these are immutable per-step points. keep_last_n <= 0 disables.
    #
    # Copied from the transducer_last.pt this same tick just wrote, rather than re-serialised. The
    # two calls saw identical state -- same step, same best_wer, same resume_count, and no RNG
    # consumed between them -- so a second torch.save produced the same ~650 MB payload twice: two
    # device->host pulls of the whole model plus both optimizers' moments, and twice the SSD write.
    # Same .tmp + os.replace as save_checkpoint, so an interrupt cannot leave a torn snapshot.
    if keep_last_n <= 0:
        return
    snap = os.path.join(ckpt_dir, f"transducer_step{step}.pt")
    tmp = snap + ".tmp"
    shutil.copyfile(source, tmp)
    os.replace(tmp, snap)
    existing = glob.glob(os.path.join(ckpt_dir, "transducer_step*.pt"))
    numbered = sorted(
        ((int(m.group(1)), p) for p in existing if (m := re.search(r"step(\d+)\.pt$", p))),
        key=lambda x: x[0],
    )
    for _, path in numbered[:-keep_last_n]:
        os.remove(path)


def _drop_snapshots_after(ckpt_dir: str, step: int, log) -> None:
    # Rolling snapshots are rotated by STEP NUMBER, so a snapshot from a run that was later rolled
    # back outranks every snapshot the replacement run writes and is never aged out. OBSERVED
    # 2026-08-05: after rolling back to step394200, the directory held 394200/399600/405000/410400
    # from the live run plus a step415800 left by the abandoned one -- and `average_checkpoints.py
    # --last-n 5` would have silently averaged that diverged checkpoint into the decode model.
    #
    # A snapshot ahead of the resume point describes a future this run did not take, so it is
    # deleted at startup rather than left to win the sort.
    for path in glob.glob(os.path.join(ckpt_dir, "transducer_step*.pt")):
        match = re.search(r"step(\d+)\.pt$", path)
        if match and int(match.group(1)) > step:
            os.remove(path)
            log.warning(f"removed {os.path.basename(path)}: ahead of the resumed step {step:,}.")


def _warm_start_encoder(model: TransducerModel, path: str, log) -> None:
    # BEST-RQ saves exactly encoder.* weights, and the transducer's encoder is byte-identical to
    # the one it pretrained. Predictor/joiner/heads start fresh.
    if not path or not os.path.isfile(path):
        log.warning(f"warm_start '{path}' absent; training encoder from scratch.")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    # The trunk normalisers are the one thing a pretrain predating `model.trunk_norm` cannot
    # supply, and they are the only thing this tolerates missing -- anything else absent is a
    # genuine mismatch between the checkpoint and this encoder, which is what strict is for.
    # They keep their `BiasNorm` init (gain 1.0), i.e. the trunk enters the transducer stage at
    # RMS 1 and finds its own level inside the window, rather than inheriting the pretrain's
    # amplitude -- which is the thing this run is trying not to inherit (CLAUDE.md pitfall 15
    # measures that handoff at trunk RMS 5.7-17.0 against 1.5-2.7 for the model that works).
    # A mid-run insertion is a different problem and wants a measured init: scripts/
    # migrate_trunk_norm.py.
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    stray = [k for k in missing if ".trunk_norm." not in k]
    if stray or unexpected:
        raise RuntimeError(f"warm start mismatch: missing {stray}, unexpected {unexpected}")
    if missing:
        log.info(f"warm start predates trunk_norm: {len(missing) // 2} norms start at gain 1.0.")
    log.info(f"warm-started encoder from {path}")


@torch.no_grad()
def greedy_transducer_decode(
    model: TransducerModel,
    memory: torch.Tensor,
    out_lengths: torch.Tensor,
    tokenizer,
) -> list[str]:
    # Standard RNN-T greedy: per encoder frame, emit non-blank tokens (advancing the predictor)
    # until blank or max_symbols, then step time. The whole batch walks the frame axis together, so
    # a symbol step is ONE predictor+joiner call at batch width and -- what actually costs -- one
    # device->host sync. Utterance-by-utterance, `int(logits.argmax())` drained the CUDA queue for
    # every (utterance, frame, symbol): ~70k stalls per dev probe, which was the whole cost of the
    # probe. Lane-wise the arithmetic is unchanged, so this is the same hypothesis as before.
    blank = get_config().model.blank_id
    max_symbols = get_config().decode.max_symbols
    device = memory.device
    batch = memory.shape[0]
    lengths = out_lengths.to(device)
    state = model.predictor.init_state(batch, device)
    prev = torch.full((batch,), blank, dtype=torch.long, device=device)
    ids: list[list[int]] = [[] for _ in range(batch)]

    for t in range(int(out_lengths.max())):
        enc_t = memory[:, t]  # [B, De]
        # Past its own length a lane is done: freezing it (rather than trimming the batch) leaves
        # its predictor state exactly where the per-utterance loop would have left it.
        active = lengths > t
        for _ in range(max_symbols):
            # `next_state` is already the context after consuming `prev`
            # (step's new_state == [state, prev][1:]) -- exactly what the NEXT step needs. Reusing
            # it (instead of re-deriving state from the just-emitted token) is both the correctness
            # requirement and avoids a second, redundant predictor forward pass.
            pred_out, next_state = model.predictor.step(state, prev)
            tok = model.joiner.step(enc_t, pred_out).argmax(dim=-1)  # [B]
            emit = active & (tok != blank)
            # One sync per symbol step, for both "did anyone emit" and "what did they emit". A lane
            # that blanks here reads the same state, prev and enc_t on the next iteration and so
            # blanks again -- frozen, exactly as the per-utterance `break` left it.
            emitted = torch.where(emit, tok, tok.new_full((), -1)).tolist()
            if all(token < 0 for token in emitted):
                break
            for lane, token in enumerate(emitted):
                if token >= 0:
                    ids[lane].append(token)
            state = torch.where(emit.unsqueeze(1), next_state, state)
            prev = torch.where(emit, tok, prev)
    return [tokenizer.decode(x) for x in ids]


def _probe_batches(n_batches: int, n_utts: int, wer_utts: int) -> set[int]:
    # Which dev batches the greedy-transducer probe decodes. `wer_utts` is a budget in utterances;
    # at the loader's mean batch size that is `k` batches, placed evenly over [0, n_batches - 1] so
    # the probe always includes both ends of the duration-sorted range.
    k = max(1, min(n_batches, round(wer_utts * n_batches / max(1, n_utts))))
    return {round(i * (n_batches - 1) / max(1, k - 1)) for i in range(k)}


@torch.no_grad()
def _dev_metrics(
    model: TransducerModel, loader, tokenizer, device: str, wer_utts: int
) -> tuple[float, float, float, float]:
    # Greedy-CTC WER + blank fraction over the WHOLE dev set (~54k words), the same greedy-CTC WER
    # re-decoded at the DEPLOYED streaming chunk, plus a greedy-transducer WER over a `wer_utts`
    # subsample (~1k words, so +-0.008 at 1 sigma -- a probe, not a criterion; checkpoint selection
    # uses the full-context ctc_wer, see run_transducer).
    #
    # The subsample spreads evenly over the loader rather than taking its head: FrameBucketSampler
    # emits duration-sorted batches, so the leading `wer_utts` utterances are the shortest clips in
    # dev -- on this manifest that head never exceeds 2.6 s against a 32.6 s corpus maximum, and
    # carries 1.3k words. `k` batches sampled across the full range instead cost the same decode
    # budget but reach the whole duration range and 3.6k words.
    #
    # The streaming pass exists because every metric above it is full-context (chunk_size=0), so a
    # run had NO visibility into the condition it is deployed in -- and the streaming gap is this
    # model's weakest axis (test-other 12.24 % streaming vs 9.17 % offline). It is deliberately
    # greedy-CTC over the full dev set rather than a second transducer decode: full-dev CTC is
    # ~54k words (sigma ~.0011) and costs one extra encoder forward with no host syncs, whereas a
    # streaming transducer probe would inherit the subsample's +-0.008 and roughly double the
    # validation decode. It is a DIAGNOSTIC of the offline->streaming delta, not a selection metric;
    # keeping selection on the full-context number leaves the LR-schedule change attributable.
    blank_id = get_config().model.blank_id
    # Base-rate frames, and a multiple of the encoder's chunk_lcm -- the same value streaming_decode
    # and evaluate.py feed the encoder, so this tracks the shipped configuration rather than one of
    # the chunk_sizes training happens to sample.
    stream_chunk = get_config().decode.chunk_size
    model.eval()
    ctc_refs, ctc_hyps, blank_frames, total_frames = [], [], 0, 0
    stream_hyps: list[str] = []
    t_refs, t_hyps = [], []
    probe_batches = _probe_batches(len(loader), len(loader.dataset), wer_utts)
    # Validation runs the compiled modules in a mode training never uses -- eval, no_grad, and fp32
    # rather than bf16 autocast -- so letting dynamo trace it would build a SECOND full set of
    # graphs (one per channel width) purely to serve ~6 s of work every val_every steps, and it is
    # what pushed BiasNorm.forward onto dynamo's per-code-object recompile limit. Forcing eager
    # here costs nothing measurable and keeps the graph budget for the training path.
    # NB: `set_stance` is only scoped when used as a context manager -- calling it bare sets the
    # stance globally and would leave the TRAINING path eager for the rest of the run.
    with torch.compiler.set_stance("force_eager"):
        for batch_idx, batch in enumerate(loader):
            feats = batch.features.to(device)
            feat_lens = batch.feature_lengths.to(device)
            memory, out_len, ctc_logits, _, _ = model(feats, feat_lens)
            # argmax on-device, transferred once: both the blank count and the greedy decode need
            # only the winning id, so moving [B, T] ids costs ~1/500th of the [B, T, V] logits.
            best = ctc_logits.argmax(dim=-1).cpu()
            lengths = out_len.cpu()
            # Same utterances, same head, chunk-masked attention -- so the delta against ctc_hyps
            # isolates what limited context costs, with the acoustic model and decode held fixed.
            _, s_len, s_ctc_logits, _, _ = model(feats, feat_lens, stream_chunk)
            stream_hyps.extend(
                ctc_greedy_decode(s_ctc_logits.argmax(dim=-1).cpu(), s_len.cpu(), tokenizer)
            )
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
    stream_ctc_wer = jiwer.wer(ctc_refs, stream_hyps)
    t_wer = jiwer.wer(t_refs[: len(t_hyps)], t_hyps[: len(t_refs)]) if t_hyps else 1.0
    return t_wer, ctc_wer, stream_ctc_wer, blank_frames / max(total_frames, 1)


def _require_cmvn(cmvn_path: str) -> str | None:
    """Resolve the CMVN path, refusing to train on unnormalised log-mel by accident.

    `ZipformerEncoder` falls back to mean 0 / std 1 when the file is absent, which is right for
    tests and for inference (a checkpoint carries the real buffers). A fresh training run that hits
    that fallback trains on raw log-mel -- mean -5.65, std 4.06 -- and no metric in the loop reports
    it. Empty path = no normalisation, deliberately.
    """
    if not cmvn_path:
        return None
    if not os.path.isfile(cmvn_path):
        raise FileNotFoundError(
            f"cmvn statistics not found: {cmvn_path}. Run scripts/compute_cmvn.py, or pass "
            'cmvn_path="" to train on unnormalised log-mel deliberately.'
        )
    return cmvn_path


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
    cmvn = _require_cmvn(cmd.cmvn_path)
    tokenizer = SentencePieceTokenizer(cmd.tokenizer_model)
    writer = SummaryWriter(cmd.log_dir)
    tr = get_config().training.transducer
    _seed_all(tr.seed)

    train_ds = LibriSpeechDataset(
        cmd.train_manifest, tokenizer, _open_cache(cmd.cache_dir, cmd.train_cache_split, log)
    )
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
    dev_ds = LibriSpeechDataset(
        cmd.dev_manifest, tokenizer, _open_cache(cmd.cache_dir, cmd.dev_cache_split, log)
    )
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

    model = TransducerModel(cmvn_path=cmvn).to(device)
    _warm_start_encoder(model, cmd.warm_start, log)
    if tr.grad_checkpoint:
        model.encoder.stacks = torch.nn.ModuleList([_Checkpointed(s) for s in model.encoder.stacks])
    # After the warm start and after any checkpoint wrapping, so the compiled leaves sit inside the
    # recompute region rather than around it (that combination is exercised and works). Parameters
    # are untouched, so this is free to sit either side of build_optimizer.
    n_compiled = compile_hot_modules(model) if tr.compile_modules else 0

    optimizers = build_optimizer(model, get_config().optim)
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    n_params = sum(p.numel() for p in model.parameters())
    console.print(
        Panel(
            f"device {device} · params {n_params / 1e6:.1f} M · steps {cmd.total_steps:,}\n"
            f"lr {tr.lr_schedule} · warmup {tr.warmup_steps:,} · stable {tr.lr_stable_ratio:.2f}"
            f" · decay {tr.lr_decay_frac:.2f} · floor {tr.lr_min_ratio:.3f}\n"
            f"chunks {tr.chunk_sizes} · ctc_aux {get_config().transducer.ctc_aux_weight}\n"
            f"compiled modules {n_compiled or 'off (eager)'}"
            f" · grad_checkpoint {tr.grad_checkpoint}\n"
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
    # The floor rides in the checkpoint so a resume inherits the run's quietest regime instead of
    # re-arming against whatever regime it resumes into.
    grad_guard = GradNormGuard(
        tr.guard_window,
        tr.guard_trend_factor,
        tr.guard_patience,
        floor=float(resumed["extra"].get("guard_norm_floor", 0.0)),
    )
    # The population of branch gains already resting on the ceiling, carried in the checkpoint for
    # the same reason the guard's floor is: a resume must not adopt the regime it resumes INTO as
    # its reference. Absent from the `extra` of a pre-2026-08-21 checkpoint, in which case the first
    # logged count latches it.
    #
    # The high-water mark rides along for the opposite reason: the count rattles (min 0 / median 2 /
    # max 5 over every 20k-step window of this run past 160k), so a mark reconstructed from the
    # baseline re-warns levels the run reported tens of thousands of steps ago. Absent from a
    # pre-2026-08-26 `extra`, in which case it falls back to the baseline, i.e. to the old
    # behaviour.
    resumed_baseline = resumed["extra"].get("gain_ceiling_baseline")
    resumed_high_water = resumed["extra"].get("gain_ceiling_high_water")
    gain_watch = GainCeilingWatch(
        baseline=None if resumed_baseline is None else int(resumed_baseline),
        high_water=None if resumed_high_water is None else int(resumed_high_water),
    )
    gain_ceiling = get_config().model.biasnorm_log_scale_max
    guard_params = guarded_parameters(model)
    gate_params = unguarded_parameters(model)
    gain_params = branch_gain_params(model)
    mix_params = stack_mix_params(model)
    n_stacks = len(mix_params)
    # Under "full" the simple term IS the rnnt term (TransducerModel.rnnt_loss returns the same
    # tensor three times) and CR-CTC is identically 0 when off, so both would log a duplicate and a
    # constant. Only report what carries information.
    log_simple = tr.rnnt_loss == "pruned"
    log_cr = get_config().transducer.cr_ctc
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")
        _drop_snapshots_after(cmd.ckpt_dir, step, log)
    train_sampler._seed = tr.seed + resume_count
    run_start = time.perf_counter()
    # Anchor the rate window at the RESUMED step, not at 0: the first log after a resume otherwise
    # divides every step of the previous run by this run's wall clock (a 2,509-step resume reported
    # 63 it/s and a 4 h ETA against a real 5.7 and 44 h).
    win_start, win_step = run_start, step
    # Kept on-device and only .item()'d at log_every: this scalar is read once per log window, so
    # casting it per optimizer step would buy a device->host sync for nothing.
    last_grad_norm = torch.zeros((), device=device)
    last_guard_norm = torch.zeros((), device=device)
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
                lr_shape = lr_at(
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
                        total, rnnt, ctc, ictc, cr, simple = model.joint_loss(batch, chunk, step)
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
                        f"CUDA OOM @ step {step:,}: batch dropped (skips {oom_skips}). "
                        "Persistent skips mean the budgets in config/training.yaml are too high "
                        "for the free VRAM."
                    )
                    continue

                accumulated += 1
                if accumulated == tr.grad_accum:
                    # Two clips, not one over model.parameters(). The gates and biases carry
                    # ~99.9 % of the global norm, so a single clip lets one scalar gate set the
                    # factor every weight matrix's gradient is scaled by -- and that factor moves
                    # ~100x step to step, which reweights Muon's momentum EMA and steers the
                    # encoder. See _train_utils.unguarded_parameters for the measured collapse.
                    #
                    # The scalars are then clipped PER TENSOR rather than against a shared norm.
                    # Splitting the clip in two alone did not remove the coupling, it moved it:
                    # encoder.stacks.3.bypass owns the scalar group's norm outright and was
                    # measured sitting at exactly grad_clip, i.e. rescaling all 414 other scalars
                    # every step. See _train_utils.clip_grads_per_tensor.
                    #
                    # The matrices keep the shared-norm clip: they are homogeneous, their norm sits
                    # ~0.9 against a bound of 5 so it never fires, and Muon renormalises its own
                    # update regardless.
                    #
                    # Order matters: the guard reads the weight matrices' norm BEFORE its clip, so
                    # it sees the true gradient rather than a rescaled one. `last_grad_norm` is the
                    # scalars' pre-clip norm, which is numerically the old global norm to 4 figures
                    # and so keeps train/grad_norm comparable with the existing run history.
                    last_guard_norm = grad_norm_of(guard_params)
                    last_grad_norm = clip_grads_per_tensor(gate_params, tr.grad_clip)
                    torch.nn.utils.clip_grad_norm_(guard_params, tr.grad_clip)
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    # Projected gradient descent for every bounded parameter -- the stacks'
                    # bypass gates and all 98 BiasNorm gains. Without it a parameter pushed past a
                    # bound leaves clamp's gradient dead zone and never trains again.
                    project_constraints(model)
                    accumulated = 0

                if step % tr.log_every == 0:
                    now = time.perf_counter()
                    its = (step - win_step) / max(now - win_start, 1e-9)
                    eta = _fmt_hms((cmd.total_steps - step) / its) if its > 0 else "-"
                    # Watch the pretrained encoder drift: a fast climb here after warmup is the
                    # erosion signature the encoder_lr_scale / per-token loss fix target.
                    # _foreach_norm folds the ~600 per-parameter reductions into one fused
                    # multi-tensor launch; the norm of the per-tensor norms is the global L2 norm.
                    with torch.no_grad():
                        enc_norm = torch.linalg.vector_norm(
                            torch.stack(torch._foreach_norm(list(model.encoder.parameters())))
                        )
                        # The largest branch gain in the model, exp(max log_scale) over all 98
                        # BiasNorms. This is the axis GradNormGuard cannot see: it reads the weight
                        # matrices, which held flat at ~0.9 through the whole 2026-08-09 divergence
                        # while this number went 1.0 -> 29.4. Bounded by
                        # model.biasnorm_log_scale_max. It coming to REST at that bound is itself
                        # the signal that the amplitude escape is being ridden -- on the 600k run
                        # it sat pinned at the old exp(2.5) = 12.18 for the last 25k steps.
                        gains = torch.stack(gain_params)
                        max_gain = gains.max().exp()
                        n_ceiling = gains_at_ceiling(gains, gain_ceiling)
                        # The other half of what a stack emits, and the half nothing used to watch:
                        # `b*g` below is the PROCESSED share, while the residual share runs
                        # through `in_proj`, which no normaliser touches. Bounded now by
                        # model.stack_in_proj_max_sigma; resting ON that bound is the signal.
                        max_trunk = trunk_gain_max(model)
                        # The half of the spectrum `max_trunk` cannot see. Pinned at the bound it
                        # is constant by construction, and the 2026-08-24 collapse ran entirely
                        # underneath it: stack 3's stable rank 48.9 -> 21.7 over 21.6k steps with
                        # sigma_1 reading exactly 10.00 at every checkpoint.
                        min_stable_rank = trunk_stable_rank_min(model)
                        # Per stack, the residual share (1 - b) and the processed share (b * g) of
                        # `(1 - b) * residual + (b * g) * x_hat`. Twelve more scalars into the same
                        # cat, so still one sync. Without these the 2026-08-09 divergence was only
                        # legible by loading five checkpoints and diffing them: `bypass` alone
                        # reads as a gate drifting, `branch_gain_max` alone says a gain is high but
                        # not which stack rode it there. See _train_utils.stack_mix_params.
                        mix = torch.cat(
                            [
                                torch.stack([1.0 - b.clamp(0.0, 1.0) for b, _ in mix_params]),
                                torch.stack([b.clamp(0.0, 1.0) * g.exp() for b, g in mix_params]),
                                torch.stack(trunk_rms_values(model)),
                            ]
                        ).float()
                        # Every .item() drains the CUDA queue, so pulling these scalars one at a
                        # time costs one pipeline stall each per log window. Stack them, sync once.
                        # .float() per element, not on the stack: under bf16 autocast the loss
                        # terms are not guaranteed to share a dtype, and torch.stack rejects a
                        # mixed-dtype list outright.
                        scalars = torch.stack(
                            [
                                t.float()
                                for t in (
                                    total,
                                    rnnt,
                                    ctc,
                                    ictc,
                                    cr,
                                    simple,
                                    last_grad_norm,
                                    last_guard_norm,
                                    enc_norm,
                                    max_gain,
                                    n_ceiling,
                                    max_trunk,
                                    min_stable_rank,
                                )
                            ]
                        )
                    flat = torch.cat([scalars, mix]).tolist()
                    (
                        v_loss,
                        v_rnnt,
                        v_ctc,
                        v_ictc,
                        v_cr,
                        v_simple,
                        v_gnorm,
                        v_guard,
                        v_enc,
                        v_gain,
                        v_ceiling,
                        v_trunk,
                        v_stable_rank,
                    ) = flat[:13]
                    v_res = flat[13 : 13 + n_stacks]
                    v_proc = flat[13 + n_stacks : 13 + 2 * n_stacks]
                    v_trunk_rms = flat[13 + 2 * n_stacks :]
                    terms = f"rnnt {v_rnnt:5.2f}"
                    if log_simple:
                        terms += f" simple {v_simple:5.2f}"
                    terms += f" ctc {v_ctc:5.2f} ictc {v_ictc:5.2f}"
                    if log_cr:
                        terms += f" cr {v_cr:5.2f}"
                    log.info(
                        f"step {step:>7,}/{cmd.total_steps:,} │ loss {v_loss:6.3f} ({terms}) │ "
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
                        ("lr", lr),
                        # Global pre-clip norm: what grad_clip acts on, and a gate-dominated
                        # number. Kept for continuity with the existing run history, but read
                        # grad_norm_guarded for model health -- that one is the weight matrices.
                        ("grad_norm", v_gnorm),
                        ("grad_norm_guarded", v_guard),
                        ("encoder_param_norm", v_enc),
                        ("branch_gain_max", v_gain),
                        # How many of the 98 gains are RESTING on the ceiling. The level of
                        # branch_gain_max cannot separate a warm start from a runaway; this count
                        # can, because the runaway spreads (1 tensor at 291.6k, 3 at 307.8k).
                        ("gains_at_ceiling", v_ceiling),
                        # Largest in_proj SPECTRAL norm, i.e. the trunk's worst-case gain per
                        # stack. Bounded by model.stack_in_proj_max_sigma -- unlike
                        # branch_gain_max there is no warm start already on the bound, so the
                        # LEVEL is the signal. A bound, though; stack_mix/*_trunk below is the
                        # amplitude the data actually gets, and that is the axis that failed.
                        ("trunk_gain_max", v_trunk),
                        # How many directions the flattest trunk operator still carries, out of
                        # its input width. `trunk_gain_max` resting on the bound is expected;
                        # THIS falling while it rests there is the projection eating the matrix.
                        ("trunk_stable_rank_min", v_stable_rank),
                    ):
                        writer.add_scalar(f"train/{name}", val, step)
                    # Only when they say something the tags above do not: `rnnt_simple` is the same
                    # tensor as `rnnt` under the "full" objective and `cr_ctc` is identically 0
                    # with CR-CTC off.
                    if log_simple:
                        # What picks the pruning band, so it diverging is a failure that would
                        # otherwise show up only as a slowly worsening WER.
                        writer.add_scalar("train/rnnt_simple", v_simple, step)
                    if log_cr:
                        writer.add_scalar("train/cr_ctc", v_cr, step)
                    for i, (res, proc) in enumerate(zip(v_res, v_proc)):
                        writer.add_scalar(f"stack_mix/{i}_residual", res, step)
                        writer.add_scalar(f"stack_mix/{i}_processed", proc, step)
                    # What `in_proj` actually emitted, per stack. The two shares above describe the
                    # PROCESSED half and read healthier at the 2026-08-22 collapse than 20k steps
                    # before it (stack 1's b*g 2.37 -> 0.73) while this went to RMS 570.
                    for i, trunk in enumerate(v_trunk_rms):
                        writer.add_scalar(f"stack_mix/{i}_trunk", trunk, step)
                    # The one observable that led the 2026-08-09 divergence, and the one
                    # GradNormGuard is structurally blind to: it reads the weight matrices, whose
                    # norm went 0.82 -> 1.60 while the global norm went 1.95 -> 201. The bound
                    # pinned 25k steps before dev WER moved.
                    #
                    # Fired on the COUNT and only on a new high-water mark, never on the level: the
                    # warm start ships gains already on the ceiling (6 of 97 in bestrq_encoder.pt),
                    # so a level test warns on step 0 of a healthy run and on all 2,400 log lines
                    # after it. What separates the two is that the population grows -- see
                    # GainCeilingWatch. A warning, not an abort: with the ceiling at exp(1.0) a
                    # pinned gain costs the next stack a factor of ~7 in the work it can do, which
                    # is a bad regime rather than a diverged one.
                    if gain_watch.update(int(v_ceiling)):
                        log.warning(
                            f"{int(v_ceiling)} branch gains now rest at the ceiling"
                            f" (exp({gain_ceiling:.1f}) = {math.exp(gain_ceiling):.2f}) @ step"
                            f" {step:,}, up from a baseline of {gain_watch.baseline}. A stack is"
                            " amplifying into the next one as hard as it is allowed."
                            " Watch dev/ctc_wer and stack_mix/*; see config/model.yaml."
                        )
                    # Not armed during warmup. The floor is a running MINIMUM, so whatever regime it
                    # first sees is the reference for the entire run -- and the window fills at
                    # window * log_every = 5,000 steps, which is inside warmup_steps = 7,500. That
                    # is the LR ramp against a fresh joiner, not the regime the run trains in, and
                    # the guard can both latch a floor there and fire on the ramp's own transient.
                    abort, gnorm_median = (
                        grad_guard.update(v_guard) if step >= tr.warmup_steps else (False, 0.0)
                    )
                    if gnorm_median > 0.0:
                        writer.add_scalar("train/guard_norm_median", gnorm_median, step)
                        writer.add_scalar("train/guard_norm_floor", grad_guard.floor, step)
                    if abort:
                        # No checkpoint written. transducer_last.pt is at most ckpt_every steps
                        # old and therefore closer to the healthy regime than this moment is;
                        # saving here would overwrite the best available resume point with the
                        # diverged state.
                        log.error(
                            f"GRADIENT RUNAWAY @ step {step:,}: weight-matrix grad-norm median"
                            f" {gnorm_median:.2f} is"
                            f" {gnorm_median / max(grad_guard.floor, 1e-9):.1f}x this run's"
                            f" quietest ({grad_guard.floor:.2f}) for {tr.guard_patience} windows."
                            " NOT checkpointing: transducer_last.pt is the healthier resume point."
                            " grad_clip cannot bound this (Newton-Schulz renormalises Muon's"
                            " update), and neither can weight decay: against a gradient this"
                            " sign-consistent its equilibrium is far outside any sane parameter"
                            " range. DIAGNOSE BEFORE ROLLING BACK: read train/trunk_gain_max and"
                            " stack_mix/*_residual first. A stack whose residual share has gone to"
                            " 0 or 1, or a trunk gain resting on model.stack_in_proj_max_sigma, is"
                            " an amplitude problem that a lower LR only slows down, and the"
                            " snapshots keep_last_n spans may all postdate it. Otherwise lower the"
                            " LR by cutting total_steps so the WSD anneal starts, and roll back to"
                            " the newest transducer_step*.pt that predates the climb."
                        )
                        writer.close()
                        return last_ckpt
                    win_start, win_step = now, step
                if step > 0 and step % tr.val_every == 0:
                    t_wer, ctc_wer, stream_ctc_wer, blank_frac = _dev_metrics(
                        model, dev_loader, tokenizer, device, tr.dev_wer_utts
                    )
                    writer.add_scalar("dev/transducer_wer", t_wer, step)
                    writer.add_scalar("dev/ctc_wer", ctc_wer, step)
                    # Full-context minus streaming on the same utterances. Watch it through the WSD
                    # anneal: the anneal sharpens the model against whatever context it is given, so
                    # this gap widening while ctc_wer falls is the signal that the LR schedule is
                    # buying offline accuracy at streaming's expense.
                    writer.add_scalar("dev/ctc_wer_stream", stream_ctc_wer, step)
                    writer.add_scalar("dev/ctc_wer_stream_gap", stream_ctc_wer - ctc_wer, step)
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
                            extra={
                                "guard_norm_floor": grad_guard.floor,
                                "gain_ceiling_baseline": gain_watch.baseline,
                                "gain_ceiling_high_water": gain_watch.high_water,
                            },
                        )
                    log.log(
                        "SUCCESS" if best else "INFO",
                        f"dev ctc-WER {ctc_wer:.4f}"
                        f"{'  ← best' if best else f'  (best {best_wer:.4f})'} │ "
                        f"stream {stream_ctc_wer:.4f} (+{stream_ctc_wer - ctc_wer:.4f}) │ "
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
                        extra={
                            "guard_norm_floor": grad_guard.floor,
                            "gain_ceiling_baseline": gain_watch.baseline,
                            "gain_ceiling_high_water": gain_watch.high_water,
                        },
                    )
                    _write_rolling_snapshot(cmd.ckpt_dir, last_ckpt, step, tr.keep_last_n)

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
                        extra={
                            "guard_norm_floor": grad_guard.floor,
                            "gain_ceiling_baseline": gain_watch.baseline,
                            "gain_ceiling_high_water": gain_watch.high_water,
                        },
                    )
                    log.warning(f"interrupt received; checkpointed @ step {step:,}, exiting.")
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
        extra={
            "guard_norm_floor": grad_guard.floor,
            "gain_ceiling_baseline": gain_watch.baseline,
            "gain_ceiling_high_water": gain_watch.high_water,
        },
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
