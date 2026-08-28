# PretrainEncoder

## Purpose
Self-supervised BEST-RQ pretraining of the Zipformer encoder: span-mask the log-mel input, predict
frozen random-projection quantizers' codes for the clean, masked positions. Warm-starts supervised
transducer training with encoder weights learned before any transcript is seen.

## Entry Point
- Type: script (`pretrain_bestrq.py` → `run_pretrain`)
- Input: `BestRqPretrainCommand` (CLI flags mirror the DTO; `--fresh` ignores `bestrq_last.pt`)
- Output: `data/checkpoints/bestrq_last.pt`, `data/checkpoints/bestrq_encoder.pt`

## Data Ownership
- Consumes artifacts: the fp16 mel cache (`data/features/mel/*`, via `FeatureCacheReader`),
  a train manifest (`data/manifests/*.jsonl`, transcripts ignored, `train_sp2` by default),
  a dev manifest for the held-out probe, `data/features/cmvn.pt` (**required**: absent is a
  `FileNotFoundError`, because the fallback would both train on raw log-mel and de-normalise the
  mask fill through mean 0 / std 1, rebuilding the +1.46 sigma plateau `BestRqMask` avoids;
  `cmvn_path=""` opts out on purpose).
- Produces artifacts: `data/checkpoints/bestrq_last.pt`, the full training state (model, optimizers,
  step and RNG) that serves as the crash/interrupt resume point, and
  `data/checkpoints/bestrq_encoder.pt`, an encoder-only state_dict (`model.encoder.state_dict()`)
  that never includes the BEST-RQ head. Talks to `TrainAcousticModel` only through the encoder
  artifact and its `encoder_init` warm-start path, never by importing internals.

## Shared Kernel
- `Config_Adapter.get_config().pretrain`: mask/codebook/schedule tunables. `.optim`: Muon/AdamW.
- `Checkpoint_Adapter.save_checkpoint`, `resume_if_available`, `SignalGuard`: SIGINT-safe
  checkpointing plus crash/interrupt resume.
- `Optimizer_Adapter.build_optimizer`: Muon + AdamW partitioning.
- `LrSchedule.lr_at`, `GradientClipping`, `ParameterProjection.project_constraints`: the same
  schedule, two-part clip and bounded-parameter projection the transducer stage runs.

## Notes

### Optimizer
The LR schedule is applied as a 0->1->0 shape multiplier against each optimizer *group's*
snapshotted peak LR, not as a single absolute value, because a uniform overwrite would clobber
Muon's much larger base LR relative to AdamW's.

`optim.encoder_lr_scale` is overridden to 1.0 for this stage. It is discriminative fine-tuning:
it protects a *warm-started* encoder while fresh predictor/joiner/heads adapt, which is a
transducer-stage concern. Here the encoder is the thing being trained, and because `BestRqModel`
names it `encoder` it matched the prefix `build_optimizer` keys on: at the configured 0.5 that ran
53.8 M of the model's 62.2 M trainable parameters at half their calibrated peak while only the
8.4 M `pred_head` (`Linear(256 -> num_codebooks * 8192)`) ran at full.

Gradients are clipped in two parts, matching `TrainAcousticModel`: a shared-norm clip over the
weight matrices, and a per-tensor clip over the scalars (biases, `BiasNorm.log_scale`, the six
`ZipformerStack.bypass` gates), which otherwise carry ~99.9 % of a single global norm and let one
gate set the rescale factor for every matrix. `project_constraints` then re-projects the bounded
parameters after each step. The v1.0 `bestrq_encoder.pt` shipped three `log_scale` at 2.0 to 2.3
against a p95 of 0.77, so the amplitude escape documented in `config/model.yaml` began in *this*
stage and was inherited by the warm start.

### Objective
`BestRqModel` runs `num_codebooks` independent quantizers (seeds `seed + i`) against one
`num_codebooks * codebook_size`-wide head, and the loss is the mean of their cross-entropies,
USM's multi-softmax. One codebook makes the whole pretraining signal a single frozen random draw;
measured across 20 seeds on 300 dev utterances, target code entropy at K=8192, D=16 ranges 7.64 to
8.90 bits.

Head and quantizers are evaluated **only on the positions that enter the loss**. Both are
8192-wide readouts and ~55 % of the grid is unmasked, so computing them on the full grid and
selecting afterwards discarded more than half of the stage's two largest GEMMs. Boolean indexing
syncs on the selected count, which is the step's one host stall and replaces the two the previous
`select.any()` + `logits[select]` pair paid.

Labels are drawn in fp32 regardless of the caller's autocast: the quantizer is an argmax over 8192
near-ties, and resolving those in bf16 makes the target of a given frame depend on numerical noise.

### Masking
Starts are sampled per frame at `mask_prob` and expanded by a left-window max over `mask_span`, so
the masked fraction is `1-(1-p)^span` and the whole batch is masked in a handful of kernels rather
than a Python loop over utterances and spans. Every utterance is guaranteed at least one span.

The fill is specified in **CMVN-normalized space** and de-normalized on the way out, because CMVN
lives inside the encoder and the mask is applied to raw log-mel. The paper's `N(0, 0.1)` follows
its "the input data is normalized to have 0 mean and standard deviation of 1"; drawing `N(0, 0.1)`
in raw log-mel instead put masked frames at +1.46 σ of the data with 0.024 σ of spread: a constant
high-energy plateau, not a removed signal.

`chunk_sizes` mirrors `training.transducer.chunk_sizes`, one sampled per batch, so the encoder meets
limited right-context here rather than first meeting it under supervision on a fifth of the data.

### Monitoring
`dev/loss` and `dev/acc` come from a fixed set of held-out batches under a pinned mask seed (the
RNG is saved and restored around the probe), so the curve reflects the encoder rather than which
frames a given probe happened to hide. `pretrain/loss` alone is not a progress signal: the
random-projection targets have an irreducible entropy floor that flattens the loss long before the
representation stops improving.
