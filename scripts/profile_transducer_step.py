# Where a transducer training step actually spends its time (GPU; user-run).
#
# A step's FLOP count says the joiner readout should dominate, but the measured step rate implies an
# effective throughput an order of magnitude under what the card reaches on large GEMMs -- i.e. the
# step is bound by kernel-launch count and memory traffic, not arithmetic. This script splits them:
# stage timings say WHERE, the kernel table plus the launch count say WHICH KIND. Run it before
# changing anything, and again after, on the same manifest so the numbers are comparable.
import argparse
import os
import statistics as st
import time
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.RnntLoss import rnnt_loss
from src.shared_kernel.RnntLossPruned import prune_ranges, rnnt_loss_pruned, rnnt_loss_simple
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand
from src.slices.TrainAcousticModel._train_utils import compile_hot_modules


def _loader(
    manifest: str, split: str, tokenizer: SentencePieceTokenizer, workers: int
) -> DataLoader:
    tr = get_config().training.transducer
    # Same cache the trainer reads, so "loader supplies Nx realtime" measures the real run's supply
    # rather than the FLAC-decode path's.
    cache_dir = get_config().features.cache_dir
    cache = (
        FeatureCacheReader(cache_dir, split)
        if os.path.isfile(os.path.join(cache_dir, f"{split}.header.json"))
        else None
    )
    return DataLoader(
        LibriSpeechDataset(manifest, tokenizer, cache),
        batch_sampler=FrameBucketSampler(
            manifest,
            tr.max_frames_per_batch,
            shuffle=True,
            seed=tr.seed,
            max_tokens_per_batch=tr.max_tokens_per_batch,
            max_lattice_per_batch=tr.max_lattice_per_batch,
            token_sort_window=tr.token_sort_window,
        ),
        collate_fn=collate_features,
        num_workers=workers,
        pin_memory=True,
    )


def _elapsed[T](fn: Callable[[], T], device: str) -> tuple[float, T]:
    # One synchronize per stage: the stages are being attributed individually, so the queue has to
    # drain before the clock stops or a stage's cost lands on whichever stage next forces a sync.
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e3, out


def _stage_breakdown(model: TransducerModel, batch, chunk: int, device: str) -> None:
    tr = get_config().training.transducer
    t = get_config().transducer
    feats = batch.features.to(device)
    flens = batch.feature_lengths.to(device)
    tokens = batch.tokens.to(device)
    tlens = batch.token_lengths.to(device)
    blank = get_config().model.blank_id
    rows = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ms, view = _elapsed(lambda: apply_spec_augment_batch(feats, flens), device)
        rows.append(("specaugment", ms))
        ms, enc = _elapsed(
            lambda: model.encoder(view, flens, chunk, return_intermediates=model.interctc_layers),
            device,
        )
        rows.append(("encoder fwd", ms))
        memory, out_len, inters, base_len = enc
        ms, ctc_logits = _elapsed(lambda: model.ctc_head(memory), device)
        rows.append(("ctc head", ms))
        ms, ictc_logits = _elapsed(
            lambda: [h(x) for h, x in zip(model.interctc_heads, inters)], device
        )
        rows.append(("interctc heads", ms))
        ms, _ = _elapsed(lambda: model.ctc_loss(ctc_logits, out_len, tokens, tlens), device)
        rows.append(("ctc loss", ms))
        ms, _ = _elapsed(lambda: model.interctc_terms(ictc_logits, base_len, tokens, tlens), device)
        rows.append(("interctc losses", ms))
        blanks = torch.full((tokens.shape[0], 1), blank, dtype=torch.long, device=device)
        pred_in = torch.cat([blanks, tokens], dim=1)
        ms, pred = _elapsed(lambda: model.predictor(pred_in), device)
        rows.append(("predictor fwd", ms))
        if tr.rnnt_loss == "full":
            ms, logits = _elapsed(lambda: model.joiner(memory, pred), device)
            rows.append(("joiner fwd (lattice)", ms))
            ms, cost = _elapsed(
                lambda: rnnt_loss(
                    logits, tokens.int(), out_len.int(), tlens.int(), blank=blank, reduction="sum"
                ),
                device,
            )
            rows.append(("rnnt loss fwd", ms))
        else:
            # Split into the four stages the pruned objective actually adds, because the whole
            # point of the change is WHERE the joiner cost went -- a single "rnnt" row would hide
            # whether the simple pass or the band scan ate the saving.
            ms, proj = _elapsed(
                lambda: (model.simple_am_proj(memory).float(), model.simple_lm_proj(pred).float()),
                device,
            )
            rows.append(("simple projections", ms))
            am, lm_proj = proj
            ms, simple_out = _elapsed(
                lambda: rnnt_loss_simple(
                    am, lm_proj, tokens.int(), out_len.int(), tlens.int(), blank=blank
                ),
                device,
            )
            rows.append(("simple loss fwd", ms))
            simple_cost, occupancy = simple_out
            s_range = min(tr.s_range, pred.shape[1])
            ms, s_begin = _elapsed(
                lambda: prune_ranges(occupancy, out_len.long(), tlens.long(), s_range), device
            )
            rows.append(("prune ranges", ms))
            idx = s_begin.unsqueeze(-1) + torch.arange(s_range, device=device)
            ms, pred_band = _elapsed(
                lambda: pred.gather(
                    1, idx.reshape(pred.shape[0], -1, 1).expand(-1, -1, pred.shape[-1])
                ).view(pred.shape[0], memory.shape[1], s_range, pred.shape[-1]),
                device,
            )
            rows.append(("predictor band gather", ms))
            ms, logits = _elapsed(lambda: model.joiner.band(memory, pred_band), device)
            rows.append(("joiner fwd (band)", ms))
            ms, cost = _elapsed(
                lambda: rnnt_loss_pruned(
                    logits,
                    tokens.int(),
                    s_begin,
                    out_len.int(),
                    tlens.int(),
                    blank=blank,
                    reduction="sum",
                ),
                device,
            )
            rows.append(("pruned loss fwd", ms))
            cost = cost + simple_cost.sum()
        total = cost / tlens.sum().clamp(min=1) + t.ctc_aux_weight * model.ctc_loss(
            ctc_logits, out_len, tokens, tlens
        )
    ms, _ = _elapsed(lambda: (total / tr.grad_accum).backward(), device)
    rows.append(("backward (all of the above)", ms))
    width = max(len(name) for name, _ in rows)
    fwd = sum(ms for name, ms in rows if not name.startswith("backward"))
    print(f"\n  stage breakdown  (B={batch.features.shape[0]} chunk={chunk})")
    for name, ms in rows:
        print(f"    {name:<{width}}  {ms:8.2f} ms")
    print(f"    {'forward subtotal':<{width}}  {fwd:8.2f} ms")


def _timed_steps(
    model: TransducerModel, loader, optimizers, steps: int, device: str, warmup: int
) -> None:
    tr = get_config().training.transducer
    times, frames, cells = [], [], []
    it = iter(loader)
    for i in range(steps + warmup):
        batch = next(it)
        chunk = 0 if i % 3 == 0 else 32
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, *_ = model.joint_loss(batch, chunk)
            (total / tr.grad_accum).backward()
        if (i + 1) % tr.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
            frames.append(int(batch.feature_lengths.sum()))
            b, t_max = batch.features.shape[0], batch.features.shape[1]
            cells.append(b * ((t_max // 2 + 1) // 2) * (int(batch.token_lengths.max()) + 1))
    hop = get_config().audio.hop_length
    audio_s = st.mean(frames) * hop / get_config().audio.sample_rate
    print(
        f"\n  {st.mean(times):.1f} ms/step (median {st.median(times):.1f}, "
        f"min {min(times):.1f})  ->  {1e3 / st.mean(times):.2f} it/s\n"
        f"  {audio_s:.0f} s audio/step  ->  {audio_s / (st.mean(times) / 1e3):.0f}x realtime\n"
        f"  mean lattice cells {st.mean(cells) / 1e3:.0f}k  "
        f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated / "
        f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved"
    )


def _kernel_table(model: TransducerModel, loader, optimizers, steps: int, device: str) -> None:
    tr = get_config().training.transducer
    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(steps):
            batch = next(it)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total, *_ = model.joint_loss(batch, 0 if i % 3 == 0 else 32)
                (total / tr.grad_accum).backward()
            # Honour grad_accum here too: Muon's Newton-Schulz is the single most expensive call in
            # the trace, and firing it every micro-batch instead of every grad_accum-th would put
            # ~4x its real share on the table and skew every percentage next to it.
            if (i + 1) % tr.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
    avgs = prof.key_averages()
    launches = sum(e.count for e in avgs if e.self_device_time_total > 0)
    device_ms = sum(e.self_device_time_total for e in avgs) / 1e3
    print(
        f"\n  {launches / steps:.0f} kernel launches/step, "
        f"{device_ms / steps:.1f} ms GPU time/step "
        f"({device_ms / launches * 1e3:.1f} us mean kernel -- under ~10 us means launch-bound)"
    )
    print(avgs.table(sort_by="self_device_time_total", row_limit=25))


def main() -> None:
    # Same allocator the trainer runs under (TransducerTrainer_Handler sets this too). Without it
    # the profiler measures a DIFFERENT allocator than the run it is sizing: the default one rounds
    # every size-varying lattice allocation up to a fresh block, so reserved runs far above
    # allocated and the peak-VRAM reading is fragmentation, not footprint. Any batch budget derived
    # from that reading is wrong. Read at the first CUDA allocation, so before .to(device) is in
    # time; setdefault still lets an explicit env override win.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Profile one transducer training step.")
    p.add_argument("--train-manifest", default=base.train_manifest)
    p.add_argument("--train-cache-split", default=base.train_cache_split)
    p.add_argument("--steps", type=int, default=20, help="timed steps (after 2 warmup)")
    p.add_argument("--profile-steps", type=int, default=6, help="steps under torch.profiler")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--rnnt-loss",
        choices=("full", "pruned"),
        default=None,
        help="override training.transducer.rnnt_loss for this run, so both objectives can be "
        "measured on the same batches in one sitting (default: the configured value)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="override training.transducer.max_frames_per_batch, to sweep the batch budget against "
        "peak VRAM without editing config between runs",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="override training.transducer.max_tokens_per_batch; it is a SUM over the batch, so it "
        "binds before the frame budget on long-transcript buckets and pins B",
    )
    p.add_argument(
        "--max-lattice",
        type=float,
        default=None,
        help="override training.transducer.max_lattice_per_batch; scale it with --max-frames or "
        "the worst-batch cap silently becomes the binding budget",
    )
    p.add_argument(
        "--token-sort-window",
        type=int,
        default=None,
        help="override training.transducer.token_sort_window, to measure how much of the lattice "
        "is U padding without editing config between runs (watch `mean lattice cells`)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="force the eager path regardless of training.transducer.compile_modules, so the "
        "compiled and eager steps can be measured back to back on the same batches",
    )
    a = p.parse_args()

    tr_cfg = get_config().training.transducer
    if a.rnnt_loss is not None:
        # Mutating the cached config is what makes the comparison apples-to-apples: the model reads
        # the objective in __init__ (it decides whether the simple projections exist at all), and
        # the sampler's budgets must stay identical across the two runs.
        tr_cfg.rnnt_loss = a.rnnt_loss
    if a.max_frames is not None:
        tr_cfg.max_frames_per_batch = a.max_frames
    if a.max_tokens is not None:
        tr_cfg.max_tokens_per_batch = a.max_tokens
    if a.max_lattice is not None:
        tr_cfg.max_lattice_per_batch = int(a.max_lattice)
    if a.token_sort_window is not None:
        tr_cfg.token_sort_window = a.token_sort_window
    print(
        f"  objective {tr_cfg.rnnt_loss} · frames {tr_cfg.max_frames_per_batch:,} · "
        f"tokens {tr_cfg.max_tokens_per_batch:,} · lattice {tr_cfg.max_lattice_per_batch:.2e} · "
        f"token_sort_window {tr_cfg.token_sort_window}"
    )

    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(base.tokenizer_model)
    loader = _loader(a.train_manifest, a.train_cache_split, tokenizer, a.workers)
    model = TransducerModel(cmvn_path=base.cmvn_path).to(a.device)
    model.train()
    # Match the trainer: the stage breakdown below is only comparable to a real run if it measures
    # the same graph the run executes. The timed loop warms up for two steps, which also absorbs
    # the one-off inductor compile.
    compiled = 0
    if tr_cfg.compile_modules and not a.no_compile:
        compiled = compile_hot_modules(model)
    print(f"  compiled modules {compiled or 'off (eager)'}")
    optimizers = build_optimizer(model, get_config().optim)

    # Loader capacity first and on its own: every later number is meaningless if the GPU is waiting
    # on features rather than the reverse.
    it = iter(loader)
    next(it)
    start, seen = time.perf_counter(), 0
    for _ in range(30):
        seen += int(next(it).feature_lengths.sum())
    hop, sr = get_config().audio.hop_length, get_config().audio.sample_rate
    supply = seen * hop / sr / (time.perf_counter() - start)
    print(f"  loader supplies {supply:.0f}x realtime ({a.workers} workers)")

    # Two steps cover the allocator and cuDNN, but inductor builds its 16 graphs lazily over the
    # first several *distinct* shape classes, so under compilation a 2-step warmup leaves compile
    # time inside the sample and reads the speedup back as a slowdown. Measured cold with 2 warmup
    # steps: 180 ms; steady state on the same batches: 153.
    _timed_steps(model, loader, optimizers, a.steps, a.device, warmup=10 if compiled else 2)
    _stage_breakdown(model, next(iter(loader)), 32, a.device)
    _kernel_table(model, loader, optimizers, a.profile_steps, a.device)


if __name__ == "__main__":
    main()
