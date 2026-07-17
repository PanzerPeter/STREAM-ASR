# TrainAcousticModel

## Purpose
The Zipformer encoder (Plan 2) plus the SP5 single-pass streaming RNN-T (transducer) head that
replaced the two-pass hybrid CTC/attention path (former Plan 3 Phase 1): a `StatelessPredictor` +
`TransducerJoiner` trained jointly with the encoder's CTC head and two InterCTC taps in **one**
training stage. There is no Stage-A/Stage-B split any more — `TransducerModel` warm-starts its
encoder from the BEST-RQ pretrain (`PretrainEncoder` slice) and trains everything else from scratch
in a single run.

## Entry Points
- Transducer training: `TransducerTrainCommand` -> `run_transducer` (`train_transducer.py`) ->
  checkpoint path (`transducer_last.pt`)
- The trainer resumes from `transducer_last.pt` on restart (SP2, via `resume_if_available`) and is
  SIGINT/SIGTERM-safe — `SignalGuard` catches the signal, finishes the in-flight step, and
  checkpoints before exiting rather than losing partial progress.
- Model: `TransducerModel(features, lengths, chunk_size=0) -> (memory, out_lengths, ctc_logits,
  interctc_logits, base_lengths)`; `joint_loss(batch, chunk_size) -> (total, rnnt, ctc, interctc)`.

## Encoder interface (frozen contract)
`ZipformerEncoder.forward(features [B,T,80], lengths, chunk_size=0, return_intermediates=[...]) ->
(memory [B,T//4,256], out_lengths, intermediates, base_lengths)`. `chunk_size` selects dynamic-chunk
masking in the self-attention (0 = full context); the transducer trainer samples `chunk_size` per
batch from `{0, 16, 32}` base-rate frames so the same weights serve both offline and streaming
inference. `return_intermediates` taps the encoder output after the listed stack indices (base
rate, ~50 Hz) for the InterCTC aux heads. This signature — including `chunk_size` and
`return_intermediates` — is the frozen contract Decode's `streaming_forward` (stateful, chunked
inference) must remain equivalent to for the aligned-frame path.

## Transducer model (`TransducerModel.py`)
- Wraps `ZipformerEncoder` + a `Linear` CTC head + `nn.ModuleList` of InterCTC `Linear` heads
  (one per `transducer.interctc_layers`) + `StatelessPredictor` + `TransducerJoiner`.
- `rnnt_loss`: blank-prefixes the token sequence, runs it through the predictor, joins against the
  encoder memory to build the full `[B, T, U+1, V]` lattice, calls `torchaudio.transforms.RNNTLoss`
  with `reduction="sum"`, then **normalises per-token** (`/ token_lengths.sum()`). This matches
  `F.ctc_loss`'s per-token `"mean"`, so all three losses share one O(1) scale and the aux weights
  are not silently ~`1/avg_tokens` weaker than nominal (the SP5 InterCTC-divergence fix).
- `ctc_loss` / `interctc_terms` / `interctc_loss`: standard CTC on the main head (25 Hz) and each
  InterCTC tap (its own `base_lengths`, ~50 Hz) — CTC is rate-agnostic so mixing rates across heads
  is fine. `interctc_terms` returns the raw per-tap losses; `interctc_loss` is their weighted sum.
- `joint_loss`: applies SpecAugment to the log-mel input when `self.training` and
  `training.transducer.spec_augment`, then `total = rnnt + ctc_aux_weight * ctc +
  Σ(interctc_weights[k] * interctc_k)`. Returns `(total, rnnt, ctc, ictc_raw)` where `ictc_raw` is
  the **raw mean** interctc across taps (a direct read on intermediate-stack CTC-decodability), not
  the weighted sum. The aux CTC head also doubles as a cheap greedy dev-WER probe.
- `StatelessPredictor.py`: icefall-style — embeds the previous non-blank token (blank-prefixed for
  the sequence start) then a depthwise causal `Conv1d` over `predictor_context` frames; no
  recurrence, so streaming state is just the last `context - 1` token ids
  (`init_state`/`step` mirror `forward` exactly). Output is `BiasNorm`-normalised.
- `TransducerJoiner.py`: additive joiner — projects encoder memory and predictor output into a
  shared `joiner_dim` space, sums, `tanh`, then reads out to `logits_width`. `forward` builds the
  full `[B, T, U', V]` grid for training; `step` evaluates one `(t, u)` cell for decoding.
- Warm-start: `_warm_start_encoder` in `TransducerTrainer_Handler.py` strict-loads `encoder.*` from
  the BEST-RQ checkpoint (`data/checkpoints/bestrq_encoder.pt` by default,
  `training.transducer.warm_start`); predictor/joiner/CTC/InterCTC heads always train from scratch.

## Data Ownership
- Consumes: `data/manifests/*.jsonl`, `data/tokenizer/bpe500.model`, `data/features/cmvn.pt`,
  `data/checkpoints/bestrq_encoder.pt` (encoder warm-start, optional but default-on)
- Produces: `data/checkpoints/transducer_last.pt` (periodic), `data/checkpoints/transducer_best.pt`
  (best dev greedy-transducer WER), `runs/transducer/` (tensorboard)

## Notes
CTC/InterCTC/transducer blank id = VOCAB_SIZE (500); logits width = 501 (shared across all heads and
the joiner's blank symbol).
bf16 autocast; trains **eager** (`compile_model=False`) — `torch.compile` hits inductor bugs on this
torch 2.11 + Blackwell build. Activation checkpointing (`_Checkpointed` wrapping each stack, in
`_train_utils.py`) is optional and **off by default**, gated by `training.transducer.grad_checkpoint`.
Transducer config lives under `training.transducer` in `config/training.yaml`:
`max_frames_per_batch 18000` (tighter than BEST-RQ pretrain's 20000 — bounded by the `B*T*(U+1)`
RNN-T joiner lattice), `max_tokens_per_batch 4000` (transcript-length budget on the same lattice),
`grad_accum 4`, `warmup_steps 10000`, `total_steps 120000`, `chunk_sizes [0, 16, 32]`,
`warm_start data/checkpoints/bestrq_encoder.pt`, `dev_wer_utts 200` (periodic greedy-transducer
WER probe size). Architecture knobs (`predictor_dim`, `predictor_context`, `joiner_dim`,
`ctc_aux_weight`, `interctc_layers`, `interctc_weights`) live in `config/transducer.yaml`, separate
from the training-loop hyperparameters.
LR peaks (`adamw_lr`/`muon_lr`) live in `config/optim.yaml`, not per-stage.
Encoder is ~53.8M params (multi-rate stacks, rotary attention), unchanged by the SP5 transducer
switch; model is ~55.3M total. Bump `encoder_dims`/`encoder_layers` in `config/model.yaml` if WER
plateaus.
`_train_utils.py` holds trainer-shared helpers (`_lr_at` cosine schedule, `_seed_all`, `_fmt_hms`,
`_Checkpointed`) used by the transducer trainer (and, structurally, any future acoustic trainer).
`CtcGreedyDecoder.py` is shared by the main CTC head's dev-WER probe.
