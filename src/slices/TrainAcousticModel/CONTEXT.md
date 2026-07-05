# TrainAcousticModel

## Purpose
The M1 Zipformer encoder + M2 CTC head (Plan 2, Stage-A CTC-only training), plus the Plan 3
Stage-B hybrid path: a U2++ bidirectional attention decoder joint-trained with CTC under
dynamic-chunk masking, warm-started from the Stage-A checkpoint.

## Entry Points
- Stage-A training: `StageATrainCommand` -> `run_stage_a` -> checkpoint path (`stage_a_last.pt`)
- Stage-B training: `StageBTrainCommand` -> `run_stage_b` (`train_stage_b.py`) -> checkpoint path
  (`stage_b_last.pt`)
- Stage-A model: `AcousticModel(features, lengths) -> (logits, out_lengths)`
- Stage-B model: `HybridCtcAttention(features, lengths, chunk_size=0) -> (ctc_logits, memory, out_lengths)`

## M1 interface (frozen contract)
`ZipformerEncoder.forward(features [B,T,80], lengths, chunk_size=0) -> (memory [B,T//4,256], out_lengths)`.
`chunk_size` selects dynamic-chunk masking in the self-attention (0 = full context, the Stage-A
default); Stage-B trains with `chunk_size` sampled per-batch from `{0, 16, 32}` base-rate frames
so the same weights serve both offline and streaming inference. This signature — including the
`chunk_size` parameter — is the frozen contract Phase 2's `streaming_forward` (stateful, chunked
inference) must remain equivalent to (spec §6).

## Stage-B hybrid model
- `HybridCtcAttention` (`HybridModel.py`): wraps `ZipformerEncoder` + a `Linear` CTC head +
  `BiTransformerDecoder`. `forward` returns `(ctc_logits, memory, out_lengths)`; `joint_loss(batch,
  chunk_size)` computes CTC loss, U2++ attention loss, and the weighted total.
- `BiTransformerDecoder` (`AttentionDecoder.py`): U2++ bidirectional decoder — a shared
  embedding/position table feeds a left (L2R, 6 layers) and a smaller right (R2L, 3 layers)
  Transformer decoder stack (8 heads, dim 512, SwiGLU FFN, BiasNorm pre-norm), cross-attending the
  projected 256-dim encoder memory. Trained by teacher forcing on `dec_in_l2r`/`dec_in_r2l` ->
  `dec_out_l2r`/`dec_out_r2l` from `FeatureCollator`; used to rescore n-best in Phase 2.
- Joint loss: `total = ctc_weight * ctc + (1 - ctc_weight) * attn` with `ctc_weight = 0.3`;
  `attn = (1 - reverse_weight) * CE_L2R + reverse_weight * CE_R2L` with `reverse_weight = 0.3`
  (i.e. 0.7 L2R / 0.3 R2L), label smoothing 0.1, `ignore_index = IGNORE_ID` on padded targets.
- Warm-start: `_warm_start` in `StageBTrainer_Handler.py` loads `encoder.*` and `ctc_head.*` from
  the Stage-A checkpoint's `AcousticModel` state dict (keys map 1:1); the decoder is always
  trained from scratch.

## Data Ownership
- Consumes: `data/manifests/*.jsonl`, `data/tokenizer/bpe500.model`, `data/features/cmvn.pt`,
  `data/checkpoints/stage_a_last.pt` (Stage-B warm-start)
- Produces: `data/checkpoints/stage_a_*.pt`, `data/checkpoints/stage_b_*.pt`, `runs/stage_a/`,
  `runs/stage_b/` (tensorboard)

## Notes
CTC blank id = VOCAB_SIZE (500); logits width = 501.
bf16 autocast on both stages; both Stage-A and Stage-B train **eager** (`compile_model=False`) —
`torch.compile` is broken on this torch 2.11 + Blackwell build. Activation checkpointing
(`_Checkpointed` wrapping each stack) is optional and **off by default**, gated by `grad_checkpoint`
in `config/training.yaml` (see CLAUDE.md pitfall #4).
Stage-B config lives under `training.stage_b` in `config/training.yaml`: `max_frames_per_batch
18000`, `lr_peak 7.5e-4`, `warmup_steps 5000`, `total_steps 80000`, `chunk_sizes [0, 16, 32]`,
`ctc_weight 0.3`, `reverse_weight 0.3`, `label_smoothing 0.1`, `warm_start
data/checkpoints/stage_a_last.pt`; escape-check gate mirrors Stage-A's blank-collapse guard.
Encoder is ~54M params (multi-rate stacks, rotary attention); bump ENCODER_DIMS/LAYERS in
`config/model.yaml` if WER plateaus.
