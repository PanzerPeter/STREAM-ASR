# PretrainEncoder

## Purpose
Self-supervised BEST-RQ pretraining of the Zipformer encoder: span-mask the log-mel input, predict a
frozen random-projection quantizer's codes for the clean, masked positions. Warm-starts supervised
transducer training with encoder weights learned before any transcript is seen.

## Entry Point
- Type: script (`pretrain_bestrq.py` → `run_pretrain`)
- Input: `BestRqPretrainCommand`
- Output: `data/checkpoints/bestrq_last.pt`, `data/checkpoints/bestrq_encoder.pt`

## Data Ownership
- Consumes artifacts: the fp16 mel cache (`data/features/mel/*`, via `FeatureCacheReader`),
  a train manifest (`data/manifests/*.jsonl`, transcripts ignored), `data/features/cmvn.pt`.
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

## Notes
The LR schedule (warmup + cosine decay) is applied as a 0->1->0 shape multiplier against each
optimizer *group's* snapshotted peak LR, not as a single absolute value, because a uniform overwrite
would clobber Muon's much larger base LR relative to AdamW's.
`BestRqModel.encoder` is the same `ZipformerEncoder` class the transducer uses, so the emitted
checkpoint loads with `strict=False` and zero unexpected keys.
