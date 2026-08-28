# TrainAcousticModel

## Purpose
Train the Zipformer encoder plus the single-pass streaming RNN-T head (a `StatelessPredictor` and a
`TransducerJoiner`, jointly with the encoder's CTC head and two InterCTC taps) in one stage.

## Entry Point
- Type: CLI (`train_transducer.py`) → `run_transducer`
- Input: `TransducerTrainCommand`
- Output: `str` (checkpoint path, `transducer_last.pt`)
- Model surface: `TransducerModel(features, lengths, chunk_size=0) -> (memory, out_lengths,
  ctc_logits, interctc_logits, base_lengths)`; `joint_loss(batch, chunk_size, step) -> (total, rnnt,
  ctc, interctc, cr, simple)`, where `cr` is the CR-CTC consistency KL (0 when `transducer.cr_ctc`
  is off) and `simple` is the pruned objective's linear-joiner loss (equal to `rnnt` under the
  `full` objective). `step` drives the prune ramp only.

The trainer resumes from `transducer_last.pt` on restart (via `resume_if_available`) and is
SIGINT/SIGTERM-safe: `SignalGuard` catches the signal, finishes the in-flight step, and checkpoints
before exiting rather than losing partial progress.

## Data Ownership
- Consumes artifacts: `data/manifests/*.jsonl` (train defaults to the paired 2× speed-perturbed
  `train_sp2.jsonl`), `data/features/mel/<split>.{f16,index.npy,header.json}` (the fp16 mmap cache,
  read via `FeatureCacheReader`; absent ⇒ warn and decode FLAC per epoch),
  `data/tokenizer/bpe500.model`, `data/features/cmvn.pt` (**required**: an absent one is a
  `FileNotFoundError`, not the encoder's mean 0 / std 1 fallback, which belongs to tests and to
  inference; `cmvn_path=""` trains unnormalised on purpose), `data/checkpoints/bestrq_encoder.pt`
  (encoder warm-start, optional but default-on). Manifest and cache split are paired by the cache
  header's `manifest_fingerprint`, so `--train-manifest` and `--train-cache-split` move together.
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
  Σ(interctc_weights[k] * interctc_k)`. Returns `(total, rnnt, ctc, ictc_raw, cr, simple)` where
  `ictc_raw` is the raw mean interctc across taps (a direct read on intermediate-stack
  CTC-decodability) rather than the weighted sum. The aux CTC head doubles as a greedy dev-WER probe
  (`CtcGreedyDecoder.py`).
- `rnnt_loss`: branches on `training.transducer.rnnt_loss`. `full` materialises the whole
  `[B,T,U+1,V]` joiner lattice; `pruned` (the default) runs `shared_kernel/RnntLossPruned`: a linear
  simple joiner (`simple_am_proj` + `simple_lm_proj`, no non-linearity, so its lattice is a
  transient) gives per-frame occupancy, `prune_ranges` turns that into an `s_range`-wide band, and
  the real joiner is evaluated only on the band via `TransducerJoiner.band`. Both terms are
  per-token; `_ramp(step)` moves them from `(simple 1.0, pruned 0.1)` to
  `(simple_loss_scale, 1.0)` over `prune_warmup_steps` loader batches. `pruned` adds two modules the
  published v1.0 checkpoints do not contain, so set `rnnt_loss: full` to load them.
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

bf16 autocast, with **selective** compilation: `_train_utils.compile_hot_modules` hands the four
elementwise leaf modules (`BiasNorm`, `TransducerJoiner`, `ConvModule`, `SwiGluFfn`) to inductor at
`dynamic=True`, gated by `training.transducer.compile_modules` (default on). Whole-model
`torch.compile` still does not work here, because it hits this torch 2.11 + Blackwell (sm_120)
build's dynamic-shape tiling assert, and compiling `ZipformerBlock` measured the same speed for 17×
the warmup, because `RotaryAttention` reaches `rotary_tables`, which calls `get_config()` and breaks
the graph. The leaves have no config call, no SDPA and no data-dependent control flow, so they trace
clean; `dynamic=True` makes the shape-varying lattice a non-issue (0 recompiles over 24 unseen batch
shapes, inside dynamo's default `cache_size_limit` of 8). `nn.Module.compile` is used rather than
wrapping, so `state_dict` keys are unchanged and the checkpoints still load into the eager
decode/eval path. Activation checkpointing (`_Checkpointed` wrapping each stack, in
`_train_utils.py`) is optional and off by default, gated by `training.transducer.grad_checkpoint`,
and composes with compilation. `_train_utils.py` holds the slice-local helpers (`_seed_all`,
`_fmt_hms`, `_Checkpointed`, `compile_hot_modules`, `GradNormGuard`, `branch_gain_params`,
`gains_at_ceiling`, `GainCeilingWatch`, `stack_mix_params`, `trunk_gain_max`,
`trunk_stable_rank_min`). `GainCeilingWatch` is the branch-gain counterpart of `GradNormGuard`: it
warns (never aborts) when more `BiasNorm` gains come to rest on `model.biasnorm_log_scale_max` than
the run started with. Like the guard's floor, its baseline rides in the checkpoint's `extra` so a
resume keeps the run's own reference, and so does its high-water mark, because the count is
instantaneous and rattles (min 0 / median 2 / max 5 over every 20k-step window of the 600k run past
160k). Persisting the baseline alone had the mark rebuilt from it on every resume, so the noise
re-crossed it and re-warned levels the run had already reported ~50k steps earlier.

`trunk_gain_max` and `trunk_rms_values` watch the other half. A stack emits
`(1 - b)*in_proj(input) + b*blocks(...)`; every `BiasNorm` sits on a branch or at a block's exit, so
`stack_mix_params` and `branch_gain_params` between them describe only the processed half. `in_proj`
is the sole operator between two stacks and carries no normaliser, so the encoder's amplitude is the
product of its gains, and left unbounded that product is what killed the 2026-08-22 run, with every
other logged scalar flat (pitfalls 13-14 in CLAUDE.md). It is bounded now by
`model.stack_in_proj_max_sigma` alongside `model.stack_bypass_min`, both projected in
`ZipformerStack.project`.

`trunk_gain_max` is the largest SPECTRAL norm, not `||W||_F/sqrt(n_out)`. The first version of that
bound was on the isotropic gain, the inflation is anisotropic, and it clipped nothing in 43k steps
while the realized trunk amplitude grew 7x. And it is still only a bound: `trunk_rms_values` reports
what `in_proj` actually emitted on the last forward (logged as `stack_mix/{i}_trunk`), which is the
quantity that failed every time. Stack 0's entry is the frontend's output, the one trunk operator no
projection reaches.

`trunk_stable_rank_min` covers the blind spot the other two leave. Once the projection binds, σ₁ is
constant by construction and `trunk_gain_max` reports nothing further. The projection itself
was a uniform rescale until 2026-08-24, which held the top direction on the ceiling while trimming
every other one on 99.2 % of steps: stack 3's stable rank went 64.4 → 21.7 over 22k steps with σ₁
reading exactly 10.00 throughout, and the realized trunk RMS doubled with it (pitfall 16 in
CLAUDE.md). `ZipformerStack.project` now deflates the top singular direction instead, which is the
minimal projection onto the same ball.

And that fix is why `model.trunk_norm` exists. With sigma_1 held and the projection minimal, the
same matrix inflated in RANK instead (stable rank 38 -> 62, ||W||_F +29 %, sigma_2/sigma_1 -> 0.994)
and the realized trunk doubled twice as fast as before. Every bound in this family constrains a
parameter; the trunk normaliser constrains the activation, which is the quantity that failed all
six times. `ZipformerStack.trunk_norm` is a `BiasNorm` on each `Linear` `in_proj`'s output with its
own, wider window (`model.trunk_norm_log_scale_max`) -- wider because the healthy trunk ledger sits
just under the BRANCH ceiling. `branch_gain_params` therefore excludes them, and
`stack_mix/{i}_trunk` now reports the POST-norm residual. Inserting them into an existing
checkpoint is `scripts/migrate_trunk_norm.py`, not a reload. Reference values and the whole amplitude ledger are in `MODEL_ARCHITECTURE.md`. The pieces the BEST-RQ pretrainer needs as well live in the Shared Kernel:
`LrSchedule.lr_at` and `GradientClipping` (`guarded_parameters`, `unguarded_parameters`,
`clip_grads_per_tensor`, `grad_norm_of`).

### Configuration
Training-loop hyperparameters live under `training.transducer` in `config/training.yaml`:
`rnnt_loss full` (`pruned` is implemented and tested but 30 % slower here; see COMMANDS.md Step 4),
`max_frames_per_batch 28000`, `max_tokens_per_batch 4400` (transcript-length budget on the same
lattice), `max_lattice_per_batch 1.1e7` (a *worst-batch* cap in `B × max(frames) × max(chars)`, since
the other two are sums and bound only the average, so it is what keeps peak VRAM off the epoch's
densest batch). All three move together: each can independently pin `B`, and holding tokens at 4000
while raising frames 32k → 45k → 58k measured B = 18, 19, 19 with identical audio per step. The
shipped triple measures B = 20, 274 s audio/step, 190.6 ms/step, 7.41 GiB allocated / 7.51 GiB
reserved, but only under `expandable_segments`, which `run_transducer` sets; on the default
allocator the same lattice reserves 9.71 GiB against 5.25 allocated and looks like a ceiling.
`token_sort_window 1` (off) is the fourth knob on the same lattice: it re-sorts by transcript length
inside a window of the duration sort, cutting `U` padding from 22.9 % of cells to 2.0 % at 1024 and
epoch lattice work by 21 %; see `config/training.yaml` and `scripts/measure_lattice_waste.py`.

### Where a step goes (profiled 2026-08-03, `scripts/profile_transducer_step.py`, B = 20)
**Eager 183.7 ms/step; compiled 160.5 ms (−12.6 %), peak VRAM 6.96 → 5.98 GiB, kernel launches
20,008 → 14,683, throughput 1491 → 1704× realtime.** Eager forward splits encoder 31.3 / joiner
lattice 11.3 / RNN-T loss 12.7 / everything else 2.8 ms, with backward 103. The loader supplies
~15,000× realtime against the
~1,500× the step consumes, so it is nowhere near the critical path. `mm` + `addmm` + `bmm` run at
roughly 42 TFLOPS, so the GEMMs are near roofline and there is no arithmetic left to recover. The
encoder's 16 blocks are only ~310 GFLOP forward, i.e. ~4× off their own GEMM roofline, so the step
is bound by elementwise traffic and launch count. **That is what compilation collects, and hand
fusion does not**: `tanh_` in place, `masked_fill_`, `torch.lerp` for the stack bypass,
`add(alpha=)` for the macaron residuals, a cached RoPE table and a shared per-stack additive
attention mask were each measured against a pinned-RNG baseline and came in between −0.5 % and
+0.7 %. Four of the six were net *losses*, and all of them are subsumed by inductor. They are not
worth re-attempting.

**Validation runs eager on purpose.** `_dev_metrics` wraps its loop in
`torch.compiler.set_stance("force_eager")`, because it exercises the compiled modules in a mode
training never uses (eval, `no_grad`, and fp32 rather than bf16 autocast), and tracing it builds a
second full set of graphs (one per channel width) to serve ~6 s of work per `val_every`. Left
unforced it pushed `BiasNorm.forward` onto dynamo's **per-code-object** recompile limit, whose
default of 8 is met exactly by four widths × {train, eval}. Exceeding that limit does not raise:
dynamo logs once and demotes the function to eager, so `compile_hot_modules` also raises the limit
to 32 for headroom. Measured after the fix: validation adds 0 graphs, 16 total, 0 recompiles, and
training holds 154.7 → 153.2 ms across a full validation. `set_stance` must be used as a context
manager, because a bare call sets it globally and would leave training eager for the whole run.

**Read the stage breakdown with care under compilation.** It synchronises after every stage, which
exposes CPU-side dynamo guard checking that the real loop hides behind GPU execution: ~144 compiled
calls per encoder forward at ~25 µs each is the entire reason "encoder fwd" reads 31.3 → 35.2 ms
compiled while the step it belongs to gets 23 ms faster. The end-to-end ms/step and the launch
count are the trustworthy numbers; per-stage rows systematically penalise the compiled path.

Two costs that look like targets and are not:

* **Muon is 10.9 ms per loader step** (32 ms per optimizer step ÷ `grad_accum`), ~6 % of the step.
  Its Newton-Schulz is 942 GFLOP, more arithmetic than the model's whole forward, running at
  29.4 TFLOPS, which is the TF32 roofline for these shapes. There is nothing to recover without
  changing `ns_steps` or dropping to bf16, and the latter was already measured to move the update
  direction ~7 %.
* **`RnntLoss.backward` spends ~15 ms** on three fp32 lattice-sized passes (`_softmax`, `mul_`, and
  the fp32→bf16 store). Doing them in bf16 would halve the traffic and is **wrong**: the blank
  gradient is `softmax[blank]·scale − blank_grad`, and those two terms converge toward each other,
  so bf16 turns the subtraction into catastrophic cancellation exactly as the model fits.

Then `grad_accum 3` (84,000 frames per update, 1.17× v1.0's proven 72,000), `warmup_steps 7500`
(= `2500 × grad_accum`, holding 2,500 optimizer updates: the transient it protects against is
measured in updates, so it does not scale with run length), `total_steps 525000`,
`chunk_sizes [0, 16, 32]`, `warm_start`, `spec_augment`, `dev_wer_utts 200` (greedy-transducer WER
probe size), `keep_last_n 5`, plus the LR shape (`lr_schedule wsd` / `lr_stable_ratio 1.0` /
`lr_decay_frac 0.25` / `lr_min_ratio 0.01`).

`total_steps` counts **loader batches**, not optimizer updates: `step += 1` sits inside
`for batch in train_loader`, outside the `grad_accum` window. 600,000 × 28,000 frames = 46,667 h =
24.1 passes over the 1,933 h `train_sp2` corpus, in 200,000 updates. v1.0's 175,000 was 9.1 passes
over 961 h, about 6 % of the icefall reference recipe and the dominant cause of its `test-other`
gap. `log_every`/`val_every`/`ckpt_every` scale with `total_steps` (÷3500, ÷90, ×0.03÷`keep_last_n`)
so a longer run neither drowns in evals nor leaves hours unrecoverable.
Architecture knobs (`predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_aux_weight`,
`interctc_layers`, `interctc_weights`) live in `config/transducer.yaml`; LR peaks
(`adamw_lr`/`muon_lr`) in `config/optim.yaml`, not per stage.

AdamW and Muon bucket parameters by peak LR (2 groups each, not one group per parameter) so the fused
multi-tensor kernels engage. Checkpoints written before that merge cannot be resumed, because
`Optimizer.load_state_dict` rejects the group-count change, so they need `--fresh`. Model weights
are unaffected, and only the moments are non-portable.

### LR schedule (`shared_kernel.LrSchedule.lr_at`)
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
