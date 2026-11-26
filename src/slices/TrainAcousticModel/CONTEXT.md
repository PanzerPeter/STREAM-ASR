# TrainAcousticModel

## Purpose
Train the S2 streaming transducer (S2 encoder + CTC head + `StatelessPredictor` + `TransducerJoiner`)
in one joint stage, from scratch. Ground truth for every operator is `MODEL_ARCHITECTURE.md`.

## Entry Point
- Type: CLI (`train_transducer.py`) → `run_transducer`
- Input: `TransducerTrainCommand` (validated in `__post_init__`): `train_sources`, a tuple of
  `TrainSource(manifest, cache_split)` pairs read side by side; CLI `--train-manifest` /
  `--train-cache-split` are repeatable and pair in order
- Output: `str` (checkpoint path, `transducer_last.pt`)
- Model surface (the artifact contract Decode, Evaluate and Demo use):
  - `TransducerModel(features, lengths, chunk_size=0, left_context=-1) -> (memory, out_lengths,
    ctc_logits)`, raw log-mel in, 12.5 Hz memory out.
  - `joint_loss(batch, chunk_size, left_context, use_band) -> LossTerms(total, rnnt, ctc,
    band_scale, band_infeasible)`; every term per token.
  - `greedy_decode(memory, lengths, max_symbols) -> (ids, emission frames)`: label-looping, exact.
  - `decode_window(features, lengths, chunk, left, tail_chunks) -> (memory, kept lengths)`: the
    batched equivalent of streaming decode.
  - `S2Encoder.init_stream(batch, left_context)` / `streaming_forward(chunk, cache, n_valid)`.

## Data Ownership
- Consumes artifacts: `data/manifests/{train_sp2,cv-train,dev,cv-dev2k,dev-other}.jsonl`, their
  mel caches
  (`data/features/mel/<split>.*`, paired by manifest fingerprint), `data/tokenizer/bpe500.model`,
  `data/features/cmvn.pt` (required; absent is a `FileNotFoundError`).
- Produces artifacts: `data/checkpoints/transducer_last.pt` (resume point), `transducer_best.pt`
  (best full dev-other streaming greedy-transducer WER), `transducer_step{N}.pt` (newest
  `keep_last_n`, averaged by `scripts/average_checkpoints.py` into `transducer_avg.pt`, which
  Decode/Evaluate/Demo load), `transducer_branch{N}.pt` at each `branch_points` step (never
  rotated), and `runs/transducer/` (tensorboard).

## Shared Kernel
- `RnntLoss.rnnt_loss` (full or banded lattice) and `RnntLoss.occupancy_band`.
- `Config_Adapter.get_config()`: `model`, `transducer`, `training.transducer`, `optim`, `decode`.
- `Optimizer_Adapter` / `Muon_Optimizer`, `GradientClipping`, `LrSchedule`, `Checkpoint_Adapter`,
  `SignalGuard`, `RoPE_Transform`, `MaskUtils`.

## Files
- `DwSepSubsampling`: x4 frontend, stride-aligned left pad of 1 per stage, weightless RMSNorm.
- `FusedBranchBlock`: gated QK-norm RoPE attention, then the conv-gated MLP on its result, two
  residual steps with their own floored gates (`gate_attn`, `gate`); W_in / W_out are one
  parameter per slice q/k/v/g/u/z and a/c so Muon orthogonalises each. Reads no config in forward.
- `S2Encoder`: the U-Net (`[4, 8, 0, 8]`: S2 is a linear 6.25 Hz path), mean fill past every row's
  length, pad to 16 mel frames, fp32 residual stream from every stack entry, streaming with a
  bounded pre-RoPE key ring.
- `StreamCache`: frontend tails + per-block conv tail and k/v/valid ring.
- `TransducerModel`: heads, objective, band, label-looping greedy, the batched streaming window.
- `TrainingLoaders`: a `ConcatDataset` of one `LibriSpeechDataset` per train source plus one
  `FrameBucketSampler` over all their manifests (same order, so the indices agree).
- `TransducerTrainer_Handler` (+ `TrainingLoaders`, `TrainingHealth`, `DevMetrics`,
  `SnapshotRotation`, `GradNormGuard`): the loop, its loaders, its logged scalars, validation,
  checkpoint files, runaway guard.

## Notes
- `total_steps` counts loader batches, not optimizer updates (`grad_accum` batches per update).
- The first `band_warmup_steps` batches run the full lattice; the band is built from the detached
  CTC blank posteriors with both repair passes, and `train/band_infeasible` must read 0.
- Validation runs eager (`force_eager`) over every dev split (clean, cv, other; the last one
  supplies the fixed probe batch): greedy CTC and label-looping greedy
  transducer WER, offline and in the deployed streaming condition (decode chunk, left context,
  completed final chunk, kept tail chunk), plus `dev/band_loss_gap` and the attention-gate
  saturation on one fixed dev-other batch.
- A padded attention query may always attend to itself: an all-masked row returns NaN from SDPA,
  and 0 × NaN would carry it into real queries through the padded frame's value.
- No parameter is projected, clamped or bounded after a step: every amplitude bound is a
  consequence of the forward pass (`MODEL_ARCHITECTURE.md` §5).
