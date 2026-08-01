# Where a transducer training step actually spends its time (GPU; user-run).
#
# A step's FLOP count says the joiner readout should dominate, but the measured step rate implies an
# effective throughput an order of magnitude under what the card reaches on large GEMMs -- i.e. the
# step is bound by kernel-launch count and memory traffic, not arithmetic. This script splits them:
# stage timings say WHERE, the kernel table plus the launch count say WHICH KIND. Run it before
# changing anything, and again after, on the same manifest so the numbers are comparable.
import argparse
import statistics as st
import time
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.RnntLoss import rnnt_loss
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.SpecAugmentBatch import apply_spec_augment_batch
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand


def _loader(manifest: str, tokenizer: SentencePieceTokenizer, workers: int) -> DataLoader:
    tr = get_config().training.transducer
    return DataLoader(
        LibriSpeechDataset(manifest, tokenizer),
        batch_sampler=FrameBucketSampler(
            manifest,
            tr.max_frames_per_batch,
            shuffle=True,
            seed=tr.seed,
            max_tokens_per_batch=tr.max_tokens_per_batch,
            max_lattice_per_batch=tr.max_lattice_per_batch,
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
        ms, logits = _elapsed(lambda: model.joiner(memory, pred), device)
        rows.append(("joiner fwd (lattice)", ms))
        ms, cost = _elapsed(
            lambda: rnnt_loss(
                logits, tokens.int(), out_len.int(), tlens.int(), blank=blank, reduction="sum"
            ),
            device,
        )
        rows.append(("rnnt loss fwd", ms))
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


def _timed_steps(model: TransducerModel, loader, optimizers, steps: int, device: str) -> None:
    tr = get_config().training.transducer
    times, frames, cells = [], [], []
    it = iter(loader)
    for i in range(steps + 2):  # first two steps carry allocator + cudnn warmup
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
        if i >= 2:
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
    base = TransducerTrainCommand()
    p = argparse.ArgumentParser(description="Profile one transducer training step.")
    p.add_argument("--train-manifest", default=base.train_manifest)
    p.add_argument("--steps", type=int, default=20, help="timed steps (after 2 warmup)")
    p.add_argument("--profile-steps", type=int, default=6, help="steps under torch.profiler")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    torch.set_float32_matmul_precision("high")
    tokenizer = SentencePieceTokenizer(base.tokenizer_model)
    loader = _loader(a.train_manifest, tokenizer, a.workers)
    model = TransducerModel(cmvn_path=base.cmvn_path).to(a.device)
    model.train()
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

    _timed_steps(model, loader, optimizers, a.steps, a.device)
    _stage_breakdown(model, next(iter(loader)), 32, a.device)
    _kernel_table(model, loader, optimizers, a.profile_steps, a.device)


if __name__ == "__main__":
    main()
