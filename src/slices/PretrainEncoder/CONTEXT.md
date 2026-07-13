# PretrainEncoder

## Purpose
Self-supervised BEST-RQ pretraining of the Zipformer encoder (SP4): span-mask the log-mel input,
predict a frozen random-projection quantizer's codes for the clean, masked positions. Warm-starts
supervised Stage-A with encoder weights learned before any transcript is seen.

## Entry Point
- Type: script (`pretrain_bestrq.py` → `run_pretrain`)
- Input: `BestRqPretrainCommand`
- Output: `data/checkpoints/bestrq_last.pt`, `data/checkpoints/bestrq_encoder.pt`

## Data Ownership
- Consumes artifacts: the SP1 fp16 mel cache (`data/features/mel/*`, via `FeatureCacheReader`),
  a train manifest (`data/manifests/*.jsonl`, transcripts ignored), `data/features/cmvn.pt`.
- Produces artifacts: `data/checkpoints/bestrq_last.pt` — full training state (model + optimizers +
  step + RNG), the crash/interrupt resume point — and `data/checkpoints/bestrq_encoder.pt`, an
  encoder-only state_dict (`model.encoder.state_dict()`), never the BEST-RQ head. Talks to
  `TrainAcousticModel` only through the encoder artifact + its `encoder_init` warm-start path —
  never by importing internals.

## Shared Kernel
- `Config_Adapter.get_config().pretrain` — mask/codebook/schedule tunables; `.optim` — Muon/AdamW.
- `Checkpoint_Adapter.save_checkpoint`, `resume_if_available`, `SignalGuard` — SIGINT-safe
  checkpointing + crash/interrupt resume (SP2 harness).
- `Optimizer_Adapter.build_optimizer` — Muon + AdamW partitioning (SP3).

## Notes
The LR schedule (warmup + cosine decay) is applied as a 0->1->0 shape multiplier against each
optimizer *group's* snapshotted peak LR, not as a single absolute value — a uniform overwrite
would clobber Muon's much larger base LR relative to AdamW's (the same defect fixed in Stage-A/B).
`BestRqModel.encoder` is the same `ZipformerEncoder` class used by `AcousticModel`, so the emitted
checkpoint loads with `strict=False` and zero unexpected keys.
