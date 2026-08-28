# BEST-RQ pretrain loop: span-mask -> encoder -> masked-prediction CE, on the resumable
# harness and the Muon+AdamW optimizer. Reads the fp16 mel cache (labels ignored). Emits a
# full-state checkpoint (bestrq_last.pt, for crash/interrupt resume) plus an encoder-only checkpoint
# (bestrq_encoder.pt) that warm-starts the transducer trainer's encoder.
import os
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.shared_kernel.BiasNorm import BiasNorm
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
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.PretrainEncoder.BestRqModel import BestRqModel
from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.MelOnlyCollator import collate_mels
from src.slices.PretrainEncoder.MelOnlyDataset import MelOnlyDataset

_DEV_MASK_SEED = 1234


def _dev_batches(cmd: BestRqPretrainCommand, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise a fixed set of held-out batches once, on the host.

    The probe has to compare across steps, so the batches themselves must not move; and at
    ~16 batches of the dev cache this is ~100 MB of pinnable host memory against a second
    DataLoader that would re-shuffle every time it was iterated.
    """
    cache = FeatureCacheReader(cmd.cache_dir, cmd.dev_cache_split)
    ds = MelOnlyDataset(cmd.dev_manifest, cache)
    sampler = FrameBucketSampler(
        cmd.dev_manifest, get_config().pretrain.max_frames_per_batch, shuffle=False
    )
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_mels, num_workers=0)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= count:
            break
    return batches


@torch.no_grad()
def _dev_probe(
    model: BestRqModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> tuple[float, float]:
    """Masked-prediction loss and accuracy on held-out audio, at full context.

    The mask is redrawn every call, so the RNG is pinned to a fixed seed and then restored: without
    that, two probes differ by which frames they happened to hide as much as by anything the
    encoder learned, and the curve is unreadable. Full context (chunk_size 0) keeps the probe
    measuring representation quality rather than the chunk size the batch happened to draw.
    """
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(_DEV_MASK_SEED)
    was_training = model.training
    model.eval()
    losses, accs = [], []
    for feats, lengths in batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss, acc = model(feats.to(device), lengths.to(device))
        losses.append(loss.float())
        accs.append(acc.float())
    model.train(was_training)
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return torch.stack(losses).mean().item(), torch.stack(accs).mean().item()


def _require_cmvn(cmvn_path: str) -> str | None:
    """Resolve the CMVN path, refusing to pretrain on unnormalised log-mel by accident.

    `ZipformerEncoder` falls back to mean 0 / std 1 when the file is absent, which is right for
    tests and for inference (a checkpoint carries the real buffers). A TRAINING run that hits that
    fallback trains on raw log-mel -- mean -5.65, std 4.06 -- and in this stage the mask fill is
    de-normalised through the same statistics, so it also re-creates the constant +1.46 sigma
    plateau `apply_span_mask` exists to avoid. This is the stage that logs no amplitude at all, so
    nothing downstream would report it either. Empty path = no normalisation, deliberately.
    """
    if not cmvn_path:
        return None
    if not os.path.isfile(cmvn_path):
        raise FileNotFoundError(
            f"cmvn statistics not found: {cmvn_path}. Run scripts/compute_cmvn.py, or pass "
            'cmvn_path="" to pretrain on unnormalised log-mel deliberately.'
        )
    return cmvn_path


def run_pretrain(cmd: BestRqPretrainCommand) -> str:
    log = configure_logging()
    cmvn = _require_cmvn(cmd.cmvn_path)
    os.makedirs(cmd.ckpt_dir, exist_ok=True)
    # The mel batches are size-varying allocations, so the default caching allocator reserves far
    # above what it allocates and reads as a VRAM ceiling that is not there. Set before the first
    # CUDA allocation or it is ignored. Same setting the transducer and LM trainers use.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = cmd.device if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    p = get_config().pretrain

    cache = FeatureCacheReader(cmd.cache_dir, cmd.cache_split)
    ds = MelOnlyDataset(cmd.train_manifest, cache)
    sampler = FrameBucketSampler(
        cmd.train_manifest,
        # BEST-RQ's own budget (config/pretrain.yaml): encoder + pred_head only, no
        # predictor/joiner, so there's no RNN-T joiner lattice (B*T*(U+1)) to cap -- unlike
        # transducer.max_frames_per_batch, which exists solely to bound that lattice.
        p.max_frames_per_batch,
        shuffle=True,
        seed=p.seed,
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate_mels, num_workers=cmd.num_workers
    )

    model = BestRqModel(cmvn_path=cmvn).to(device)
    # encoder_lr_scale is discriminative fine-tuning: it protects a BEST-RQ-pretrained encoder from
    # the fresh predictor/joiner/head gradients of the SUPERVISED stage. There is no pretrained
    # encoder to protect here -- the encoder IS this stage -- and `BestRqModel.encoder` matches the
    # `encoder.` prefix build_optimizer keys on, so leaving it at the configured 0.5 ran 53.8 M of
    # the model's 62.2 M trainable parameters at half their calibrated peak LR while only the
    # 8.4 M pred_head (256 x 4*8192, one slice per codebook) ran at full. That is not a tuning
    # choice, it is the transducer's knob reaching into the wrong stage. `pretrain.lr_scale` is
    # this stage's own multiplier and replaces it -- see config/pretrain.yaml for why the
    # encoder's effective LR is not free to move.
    optimizers = build_optimizer(
        model, get_config().optim.model_copy(update={"encoder_lr_scale": 1.0})
    )
    # build_optimizer sets each group's lr to its calibrated PEAK (Muon >> AdamW per
    # config/optim.yaml). Snapshot the peaks so the schedule is applied as a 0->1->0 SHAPE
    # multiplier per group. A single absolute overwrite would clobber Muon's much larger base LR,
    # defeating the whole optimizer split.
    peak_lrs = [[g["lr"] * p.lr_scale for g in opt.param_groups] for opt in optimizers]
    guard_params = guarded_parameters(model)
    gate_params = unguarded_parameters(model)
    gain_params: list[torch.Tensor] = [
        m.log_scale for m in model.modules() if isinstance(m, BiasNorm)
    ]

    last_ckpt = os.path.join(cmd.ckpt_dir, "bestrq_last.pt")
    encoder_ckpt = os.path.join(cmd.ckpt_dir, "bestrq_encoder.pt")
    # Restore full training state (model + optimizers + step + RNG) after a crash/interrupt and bump
    # resume_count so the sampler reseeds a fresh, non-repeating epoch.
    resumed = resume_if_available(last_ckpt, model, optimizers, cmd.resume)
    step = int(resumed["step"])
    resume_count = int(resumed["resume_count"])
    sampler._seed = p.seed + resume_count  # read at FrameBucketSampler.__iter__
    if step > 0:
        log.info(f"resumed from {last_ckpt} @ step {step:,} (resume #{resume_count})")

    dev_batches = _dev_batches(cmd, p.dev_batches) if p.dev_every > 0 else []
    chunk_rng = random.Random(p.seed + resume_count)

    writer = SummaryWriter(cmd.log_dir)
    model.train()
    log.info(
        f"BEST-RQ pretrain on <{device}>: target {cmd.total_steps:,} steps, "
        f"{p.num_codebooks} codebooks, mask {p.mask_prob}x{p.mask_span}"
    )
    with SignalGuard() as guard:
        while step < cmd.total_steps:
            for feats, lengths in loader:
                lr_shape = lr_at(
                    step,
                    1.0,
                    p.warmup_steps,
                    cmd.total_steps,
                    schedule=p.lr_schedule,
                    decay_frac=p.lr_decay_frac,
                    min_ratio=p.lr_min_ratio,
                )
                for opt, peaks in zip(optimizers, peak_lrs):
                    for g, peak in zip(opt.param_groups, peaks):
                        g["lr"] = peak * lr_shape
                # One chunk size per batch, as the transducer stage does, so the encoder meets
                # limited right-context here rather than first meeting it under supervision on a
                # fifth of the data. 0 (full context) stays in the pool.
                chunk_size = chunk_rng.choice(p.chunk_sizes)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    loss, accuracy = model(feats.to(device), lengths.to(device), chunk_size)
                loss.backward()
                # Two clips, not one over model.parameters(). The scalar gates and biases carry
                # ~99.9 % of the global norm, so a single clip lets one gate set the factor every
                # weight matrix's gradient is scaled by -- and that factor moves ~100x step to
                # step, which reweights Muon's momentum EMA and steers the encoder. The scalars are
                # then clipped PER TENSOR, because splitting the clip in two alone only relocates
                # the coupling into the scalar group. See shared_kernel/GradientClipping.py.
                guard_norm = grad_norm_of(guard_params)
                grad_norm = clip_grads_per_tensor(gate_params, p.grad_clip)
                torch.nn.utils.clip_grad_norm_(guard_params, p.grad_clip)
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                # The bypass gates and BiasNorm gains are bounded, and clamp's gradient is dead
                # outside the bounds, so a parameter pushed out of range never trains again -- and
                # this stage's encoder is what the transducer warm-starts from. The unbounded gain
                # drifted here too: the v1.0 bestrq_encoder.pt carried three log_scales at 2.0-2.3
                # against a p95 of 0.77, so the warm start was already part-way into the escape.
                project_constraints(model)
                step += 1
                if step % p.log_every == 0:
                    with torch.no_grad():
                        # exp(max log_scale) over every BiasNorm. This is the axis a gradient-norm
                        # guard is structurally blind to, and it coming to REST at
                        # model.biasnorm_log_scale_max is itself the failure signal. One stack of
                        # scalars, one sync, instead of one pipeline stall per .item().
                        stats = torch.stack(
                            [
                                loss.float(),
                                accuracy.float(),
                                grad_norm.float(),
                                guard_norm.float(),
                                torch.stack(gain_params).max().exp().float(),
                            ]
                        ).tolist()
                    loss_val, acc_val, gn, gn_guarded, max_gain = stats
                    writer.add_scalar("pretrain/loss", loss_val, step)
                    writer.add_scalar("pretrain/acc", acc_val, step)
                    writer.add_scalar("train/grad_norm", gn, step)
                    writer.add_scalar("train/grad_norm_guarded", gn_guarded, step)
                    writer.add_scalar("train/branch_gain_max", max_gain, step)
                    # build_optimizer returns AdamW last ([muon, adamw] or [adamw]), so
                    # optimizers[-1] is always the AdamW (representative LR for logs).
                    writer.add_scalar("pretrain/lr", optimizers[-1].param_groups[0]["lr"], step)
                    log.info(
                        f"step {step:,}/{cmd.total_steps:,} loss {loss_val:.4f} "
                        f"acc {acc_val:.4f} gain {max_gain:.2f}"
                    )
                if dev_batches and step % p.dev_every == 0:
                    dev_loss, dev_acc = _dev_probe(model, dev_batches, device)
                    writer.add_scalar("dev/loss", dev_loss, step)
                    writer.add_scalar("dev/acc", dev_acc, step)
                    log.info(f"step {step:,} dev loss {dev_loss:.4f} acc {dev_acc:.4f}")
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
    # (drop the BEST-RQ head) for supervised transducer training.
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
