# TrainAcousticModel

## Purpose
Train the Zipformer encoder plus the single-pass streaming RNN-T head (a `StatelessPredictor` and a
`TransducerJoiner`, jointly with the encoder's CTC head and two InterCTC taps) in one stage.

## Entry Point
- Type: CLI (`train_transducer.py`) → `run_transducer`
- Input: `TransducerTrainCommand`
- Output: `str` (checkpoint path, `transducer_last.pt`)
- Model surface: `TransducerModel(features, lengths, chunk_size=0) -> (memory, out_lengths,
  ctc_logits, interctc_logits, base_lengths)`; `joint_loss(batch, chunk_size) -> (total, rnnt, ctc,
  interctc, cr)`, where `cr` is the CR-CTC consistency KL (0 when `transducer.cr_ctc` is off).

The trainer resumes from `transducer_last.pt` on restart (via `resume_if_available`) and is
SIGINT/SIGTERM-safe: `SignalGuard` catches the signal, finishes the in-flight step, and checkpoints
before exiting rather than losing partial progress.

## Data Ownership
- Consumes artifacts: `data/manifests/*.jsonl`, `data/tokenizer/bpe500.model`,
  `data/features/cmvn.pt`, `data/checkpoints/bestrq_encoder.pt` (encoder warm-start, optional but
  default-on).
- Produces artifacts: `data/checkpoints/transducer_last.pt` (periodic resume point),
  `transducer_best.pt` (best full-dev greedy-CTC WER, which is the selection metric; the
  greedy-transducer number logged beside it is a ~1k-word probe, too noisy to select on),
  `transducer_step{N}.pt` (rolling snapshots, newest `keep_last_n` kept; averaging that tail into
  `transducer_avg.pt` via `scripts/average_checkpoints.py` is a required post-training step, and
  `_avg` is what Decode/Evaluate/Demo load by default rather than `_best`), and `runs/transducer/`
  (tensorboard).

## Shared Kernel
- `RnntLoss.rnnt_loss`: the RNN-T forward-backward (this repo's own Graves kernel).
- `Config_Adapter.get_config()`: `training.transducer`, `transducer`, `model`, `optim`.
- `Optimizer_Adapter.build_optimizer` / `Muon_Optimizer`: Muon + AdamW partitioning.
- `Checkpoint_Adapter`, `SignalGuard`: atomic checkpointing plus interrupt-safe resume.
- `BiasNorm`, `SwiGluFfn`, `RoPE_Transform`, `MaskUtils`: encoder and predictor blocks.

## Encoder interface (frozen contract)
`ZipformerEncoder.forward(features [B,T,80], lengths, chunk_size=0, return_intermediates=[...]) ->
(memory [B,T//4,256], out_lengths, intermediates, base_lengths)`. `chunk_size` selects dynamic-chunk
masking in the self-attention (0 = full context); the trainer samples `chunk_size` per batch from
`{0, 16, 32}` base-rate frames so the same weights serve both offline and streaming inference.
`return_intermediates` taps the encoder output after the listed stack indices (base rate, ~50 Hz) for
the InterCTC aux heads. This signature, including `chunk_size` and `return_intermediates`, is the
frozen contract Decode's `streaming_forward` (stateful, chunked inference) must remain equivalent to
for the aligned-frame path.

## Notes

### Model composition (`TransducerModel.py`)
- Wraps `ZipformerEncoder` + a `Linear` CTC head + an `nn.ModuleList` of InterCTC `Linear` heads (one
  per `transducer.interctc_layers`) + `StatelessPredictor` + `TransducerJoiner`.
- `rnnt_loss`: blank-prefixes the token sequence, runs it through the predictor, joins against the
  encoder memory to build the full `[B, T, U+1, V]` lattice, calls `shared_kernel/RnntLoss.rnnt_loss`
  (this repo's own Graves forward-backward, not `torchaudio.transforms.RNNTLoss`) with
  `reduction="sum"`, then normalises per-token (`/ token_lengths.sum()`). This matches
  `F.ctc_loss`'s per-token `"mean"`, so all three losses share one O(1) scale and the aux weights are
  not silently ~`1/avg_tokens` weaker than nominal, an imbalance that makes InterCTC diverge.
- `ctc_loss` / `interctc_terms` / `interctc_loss`: standard CTC on the main head (25 Hz) and each
  InterCTC tap (its own `base_lengths`, ~50 Hz). CTC is rate-agnostic, so mixing rates across heads
  is fine. `interctc_terms` returns the raw per-tap losses; `interctc_loss` is their weighted sum.
- `joint_loss`: applies SpecAugment to the log-mel input when `self.training` and
  `training.transducer.spec_augment`, then `total = rnnt + ctc_aux_weight * ctc +
  Σ(interctc_weights[k] * interctc_k)`. Returns `(total, rnnt, ctc, ictc_raw, cr)` where `ictc_raw`
  is the raw mean interctc across taps (a direct read on intermediate-stack CTC-decodability) rather
  than the weighted sum. The aux CTC head doubles as a greedy dev-WER probe (`CtcGreedyDecoder.py`).
- `StatelessPredictor.py`: icefall-style. It embeds the previous non-blank token (blank-prefixed for
  the sequence start), then applies a depthwise causal `Conv1d` over `predictor_context` frames.
  There is no recurrence, so streaming state is just the last `context - 1` token ids
  (`init_state`/`step` mirror `forward` exactly). Output is `BiasNorm`-normalised.
- `TransducerJoiner.py`: an additive joiner. It projects encoder memory and predictor output into a
  shared `joiner_dim` space, sums, applies `tanh`, then reads out to `logits_width`. `forward` builds
  the full `[B, T, U', V]` grid for training; `step` evaluates one `(t, u)` cell for decoding.
- **Aligned readout.** The readout is the only GEMM over the whole lattice, and `logits_width = 501`
  is not 16-byte aligned in bf16, so cuBLAS picks an alignment-2 fallback. `forward` pads it to a
  multiple of 8 with a `-inf` bias on the pad columns: `exp()` of those logits is exactly 0, so every
  log-softmax, gather and gradient downstream is the 501-wide result and the pad columns' own
  gradient is exactly 0. Parameters stay 501-wide (checkpoints unaffected); `step` is left unpadded.
- Warm-start: `_warm_start_encoder` in `TransducerTrainer_Handler.py` strict-loads `encoder.*` from
  the BEST-RQ checkpoint (`data/checkpoints/bestrq_encoder.pt` by default,
  `training.transducer.warm_start`); predictor/joiner/CTC/InterCTC heads always train from scratch.
- **CR-CTC** (`transducer.cr_ctc`, default off, because a full run landed 0.0674 dev transducer-WER
  against the shipped recipe's 0.0553; see `config/transducer.yaml`): when on, train-mode
  `joint_loss` takes the two-view path (`_joint_loss_cr`), which builds two independently
  SpecAugment-masked views, a second CTC-only encoder forward on view 2, and a bidirectional-KL
  consistency term (`_cr_consistency`, weight `cr_weight`) tying the two CTC heads. RNN-T runs on
  view 1 only, since its `[B,T,U+1,V]` lattice is the memory hog, and the CTC term averages both
  views. The second encoder forward roughly doubles encoder activation memory, so on 12 GB pair it
  with `grad_checkpoint: true` or a lower `max_frames_per_batch`. Off restores the proven
  single-view objective and `cr == 0`.

### Shapes and execution
CTC/InterCTC/transducer blank id = `VOCAB_SIZE` (500); logits width = 501, shared across all heads
and the joiner's blank symbol. Encoder is ~53.8 M params (multi-rate stacks, rotary attention), and
the model is ~55.3 M total. Bump `encoder_dims`/`encoder_layers` in `config/model.yaml` if WER
plateaus.

bf16 autocast, and it trains eager, because `torch.compile` is not wired in. Its inductor backend
hits Triton bugs on this torch 2.11 + Blackwell (sm_120) build (partitioner crash under activation
checkpointing, and a dynamic-shape tiling assert), and the shape-varying RNN-T lattice means every
bucket would retrigger a recompile. Activation checkpointing (`_Checkpointed` wrapping each stack,
in `_train_utils.py`) is optional and off by default, gated by
`training.transducer.grad_checkpoint`. `_train_utils.py` holds the trainer-shared helpers (`_lr_at`,
`_seed_all`, `_fmt_hms`, `_Checkpointed`).

### Configuration
Training-loop hyperparameters live under `training.transducer` in `config/training.yaml`:
`max_frames_per_batch 18000` (tighter than BEST-RQ pretrain's 20000, bounded by the `B*T*(U+1)`
RNN-T lattice), `max_tokens_per_batch 4000` (transcript-length budget on the same lattice),
`max_lattice_per_batch 6.0e6` (a *worst-batch* cap in `B × max(frames) × max(chars)`, since the other
two are sums and bound only the average, so it is what keeps peak VRAM off the epoch's densest
batch), `token_sort_window 1` (off; see that key's comment for the measured trade), `grad_accum 4`,
`warmup_steps 10000`, `total_steps 175000`, `chunk_sizes [0, 16, 32]`, `warm_start`, `spec_augment`,
`dev_wer_utts 200` (greedy-transducer WER probe size), `keep_last_n 5`, plus the LR shape
(`lr_schedule wsd` / `lr_stable_ratio 1.0` / `lr_decay_frac 0.25` / `lr_min_ratio 0.01`).
Architecture knobs (`predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_aux_weight`,
`interctc_layers`, `interctc_weights`) live in `config/transducer.yaml`; LR peaks
(`adamw_lr`/`muon_lr`) in `config/optim.yaml`, not per stage.

AdamW and Muon bucket parameters by peak LR (2 groups each, not one group per parameter) so the fused
multi-tensor kernels engage. Checkpoints written before that merge cannot be resumed, because
`Optimizer.load_state_dict` rejects the group-count change, so they need `--fresh`. Model weights
are unaffected, and only the moments are non-portable.

### LR schedule (`_lr_at`)
Linear warmup, then `lr_schedule` selects the shape. `wsd` (default) holds `lr_stable_ratio * peak`
until the last `lr_decay_frac` of `total_steps`, then anneals with a 1-sqrt profile to
`lr_min_ratio * peak`; `cosine` anneals from the end of warmup to the same floor. Both land exactly
at `total_steps`, and `lr_min_ratio > 0` stops the tail steps from running at an LR that no longer
trains.

Under cosine, `total_steps` *is* the curve, not just the budget: raising it mid-run re-heats the LR
at the resume step. The 3.55 %-WER checkpoint that preceded the current one came out of exactly
that. Cosine annealed to 0 at 120k, `total_steps` was bumped to 175k, and the run resumed at 25 % of
peak and annealed a second time, giving an unintended two-cycle schedule. That is why a single 175k
cosine sits ~2 WER points behind it at a *matched* step (at 92k: 50 % of peak vs that run's 15 %).
WSD makes the same bump move only the decay window, and a fresh 175k WSD run at
`lr_stable_ratio: 1.0` landed within ~1 σ of the accidental two-cycle result (test-clean beam
4.38 %/5.84 % vs 4.30 %/5.95 %, offline/streaming), so the two-cycle schedule was luck rather than a
recipe worth reproducing.

When switching an in-flight cosine run over to WSD, set `lr_stable_ratio` to the cosine shape value
at the resume step (`0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))`) so the restart
continues at the LR that run had already reached instead of re-heating the encoder.
