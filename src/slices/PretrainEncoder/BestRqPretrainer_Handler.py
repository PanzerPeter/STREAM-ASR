# src/slices/PretrainEncoder/BestRqPretrainer_Handler.py
# BEST-RQ pretrain loop (SP4): span-mask -> encoder -> masked-prediction CE, on the SP2 resumable
# harness and the SP3 Muon+muP optimizer. Reads the SP1 fp16 mel cache (labels ignored). Emits a
# full-state checkpoint (bestrq_last.pt, for crash/interrupt resume) plus an encoder-only checkpoint
# (bestrq_encoder.pt) that warm-starts supervised Stage-A.
import math
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.shared_kernel.Checkpoint_Adapter import resume_if_available, save_checkpoint
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Optimizer_Adapter import build_optimizer
from src.shared_kernel.SignalGuard import SignalGuard
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.PretrainEncoder.BestRqModel import BestRqModel
from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.MelOnlyCollator import collate_mels
from src.slices.PretrainEncoder.MelOnlyDataset import MelOnlyDataset


def _lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def run_pretrain(cmd: BestRqPretrainCommand) -> str:
    log = configure_logging()
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    p = get_config().pretrain
    optim_cfg = get_config().optim

    cache = FeatureCacheReader(cmd.cache_dir, cmd.cache_split)
    ds = MelOnlyDataset(cmd.train_manifest, cache)
    sampler = FrameBucketSampler(
        cmd.train_manifest,
        get_config().training.stage_a.max_frames_per_batch,
        shuffle=True,
        seed=p.seed,
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate_mels, num_workers=cmd.num_workers
    )

    cmvn = cmd.cmvn_path if os.path.isfile(cmd.cmvn_path) else None
    model = BestRqModel(cmvn_path=cmvn).to(device)
    optimizers = build_optimizer(model, optim_cfg)
    # build_optimizer sets each group's lr to its calibrated PEAK (Muon >> AdamW per
    # config/optim.yaml). Snapshot the peaks so the warmup+cosine schedule is applied as a
    # 0->1->0 SHAPE multiplier per group. A single absolute overwrite would clobber Muon's much
    # larger base LR and any muP ratios, defeating SP3 (the same bug fixed in Stage-A/B).
    peak_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    last_ckpt = os.path.join(cmd.ckpt_dir, "bestrq_last.pt")
    encoder_ckpt = os.path.join(cmd.ckpt_dir, "bestrq_encoder.pt")
    # Restore full training state (model + optimizers + step + RNG) after a crash/interrupt and bump
    # resume_count so the sampler reseeds a fresh, non-repeating epoch (SP2 resumable harness).
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step = int(resumed["step"])
    resume_count = int(resumed["resume_count"])
    sampler._seed = p.seed + resume_count  # read at FrameBucketSampler.__iter__
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")

    writer = SummaryWriter(cmd.log_dir)
    model.train()
    log.info(f"BEST-RQ pretrain on <{device}> — target {cmd.total_steps:,} steps")
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for feats, lengths in loader:
                lr_shape = _lr_at(step, 1.0, p.warmup_steps, cmd.total_steps)  # 0->1->0 shape
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    loss = model(feats.to(device), lengths.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                if step % p.log_every == 0:
                    loss_val = loss.item()
                    writer.add_scalar("pretrain/loss", loss_val, step)
                    # build_optimizer returns AdamW last ([muon, adamw] or [adamw]), so
                    # optimizers[-1] is always the AdamW (representative LR for logs).
                    writer.add_scalar("pretrain/lr", optimizers[-1].param_groups[0]["lr"], step)
                    log.info(f"step {step:,}/{cmd.total_steps:,} loss {loss_val:.4f}")
                if step % p.save_every == 0:
                    save_checkpoint(
                        last_ckpt,
                        model,
                        optimizers,
                        step,
                        resume_count=resume_count,
                        kind="bestrq",
                    )
                stop = guard.stop_requested or (
                    cmd.max_steps_smoke is not None and step >= cmd.max_steps_smoke
                )
                if stop or step >= cmd.total_steps:
                    break
            if guard.stop_requested or (
                cmd.max_steps_smoke is not None and step >= cmd.max_steps_smoke
            ):
                break

    # Persist the final full-state resume point, then emit the encoder-only warm-start artifact
    # (drop the BEST-RQ head) for supervised Stage-A.
    save_checkpoint(last_ckpt, model, optimizers, step, resume_count=resume_count, kind="bestrq")
    save_checkpoint(
        encoder_ckpt,
        model.encoder,
        [],
        step,
        kind="bestrq",
        extra={"quantizer_seed": p.seed},
    )
    writer.close()
    log.info(f"pretrain done @ step {step:,} -> {encoder_ckpt}")
    return encoder_ckpt
