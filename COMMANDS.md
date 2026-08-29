# STREAM ASR command reference

Operational companion to [README.md](README.md): every command, its flags, and the failure modes
worth knowing before you hit them. Headline results and architecture live in the README.

> **Conventions.** Commands assume the repository root as the working directory with the venv
> activated (`source .venv/bin/activate`), so `python` is the 3.12 venv interpreter. Otherwise spell
> it `.venv/bin/python`. `python -m …` and `python -c …` already have the repo root on `sys.path`;
> only `scripts/*.py`, `pytest` and `mypy` need the `PYTHONPATH=.` prefix.

## 1. Pipeline at a glance

```bash
# Step 1: environment
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python scripts/verify_env.py

# Step 2: data foundation (960 h)
PYTHONPATH=. python scripts/build_manifests.py               # all 5 splits, parallel probe
PYTHONPATH=. python scripts/train_tokenizer.py               # BPE-500 over the 960 h train transcripts
PYTHONPATH=. python scripts/compute_cmvn.py                  # CMVN over a 15 % sample
PYTHONPATH=. python scripts/build_speed_perturb_manifest.py  # paired 2x train manifest -> train_sp2.jsonl
PYTHONPATH=. python scripts/precompute_features.py           # fp16 log-mel cache (~53 GB/copy, one-time)

# Step 3: self-supervised encoder pretrain
python -m src.slices.PretrainEncoder.pretrain_bestrq

# Step 4: transducer training (single joint stage, all defaults)
python -m src.slices.TrainAcousticModel.train_transducer
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.slices.TrainAcousticModel.train_transducer
tensorboard --logdir runs/transducer

# Step 5: average the tail of the rolling snapshots into the decode checkpoint (REQUIRED)
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5   # -> data/checkpoints/transducer_avg.pt

# Step 6: language model
python scripts/download_lm_text.py && gunzip -k data/lm_text/librispeech-lm-norm.txt.gz
python -m src.slices.TrainLanguageModel.train_lm             # after the prep snippet in §2, Step 6
tensorboard --logdir runs/lm

# Step 7: decode a single file
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac --offline
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac

# Step 8: corpus evaluation (tunes on the split's dev, reports on its test)
python -m src.slices.Evaluate.evaluate --clean
python -m src.slices.Evaluate.evaluate --other

# Step 9: local web demo at http://127.0.0.1:8000
python -m src.slices.Demo.serve_demo --lm-weight 0.7 --ilm-weight 0.4
```

---

## 2. Step details

### Step 1: environment

* Verification gate: expect `OK: ... cap=(12, 0)` (Blackwell sm_120).
* Hardware target: RTX 5070 / CUDA 12.8 wheels via the extra index in `requirements.txt`.

### Step 2: data foundation

* Builds 5-split JSONL manifests over `train-clean-100 + train-clean-360 + train-other-500`:
  train 281,241 / dev-clean 2,703 / dev-other 2,864 / test-clean 2,620 / test-other 2,939
  utterances.
* Trains a BPE-500 SentencePiece tokenizer into `data/tokenizer/bpe500.{model,vocab}`.
* Computes global 80-bin mean/std CMVN over a 15 % train sample into `data/features/cmvn.pt`
  (statistics converge well before the full set).
* Precomputes `data/features/mel/<split>.{f16,index.npy,header.json}`, streamed via mmap so the
  training epoch loop is GPU-bound and never re-decodes FLAC or recomputes an FFT per epoch.
  Re-run the same command after an interruption: a split whose header, index and `.f16` byte
  count agree is skipped, so only the split that died is redone (no mid-split resume). Delete a
  `<split>.header.json` to force that split to rebuild.
* The LM (Step 6) is tokenizer-specific, so regenerate its packed data and checkpoint whenever the
  tokenizer changes.

`scripts/build_speed_perturb_manifest.py` writes `train_sp2.jsonl`: every train utterance twice,
once untouched and once at **one** factor drawn per utterance from 0.9 / 1.1, with corrected
`num_samples`. 2×, not icefall's 3× cross product, because the binding constraint is disk: each
copy of the corpus is another ~53 GB of fp16 mel. The factor is `sha1(uttid, seed)`, never the
builtin `hash()`, so a rebuild is byte-identical.

The perturbation is applied at **extraction** time and baked into the `train_sp2` cache, so a
perturbed run still reads mmap slices instead of decoding FLAC per row. That makes the manifest half
the data structure: the cache is a flat memmap indexed by row order, so pairing it with a different
manifest reads the right row count and the right shapes and trains on the wrong audio. Every cache
header now carries a `manifest_fingerprint` (sha1 over each row's `uttid` and `speed`), checked by
`FeatureCacheReader` and `LibriSpeechDataset`, which turns that into a load-time error. Caches built
before the fingerprint existed have none, and are accepted unchanged, so no rebuild is needed.

`train_sp2.jsonl` + the `train_sp2` cache split are the transducer trainer's **defaults** (Step 4).
Pass `--train-manifest data/manifests/train.jsonl --train-cache-split train` to train on the clean
961 h corpus instead; the two flags must move together or the fingerprint check rejects the pair.

### Step 3: BEST-RQ encoder pretrain

Self-supervised masked-prediction pretraining of the Zipformer encoder on the mel cache
(transcripts ignored). `num_codebooks` frozen random-projection quantizers turn clean mel into
discrete targets, and the encoder predicts them from span-masked input. Long GPU job.

```bash
python -m src.slices.PretrainEncoder.pretrain_bestrq
tensorboard --logdir runs/bestrq
```

* Produces `data/checkpoints/bestrq_encoder.pt` (encoder-only warm-start artifact) and
  `bestrq_last.pt` (full-state resume point).
* The transducer trainer consumes the former by default (`training.transducer.warm_start`), so no
  extra flag is needed.
* Defaults to the 2x speed-perturbed `train_sp2` corpus, the same one the transducer stage uses.
  Pass `--train-manifest data/manifests/train.jsonl --cache-split train` for the clean 961 h set;
  the two flags move together or the cache fingerprint check rejects the pair.
* Pretrain knobs (codebooks, mask, chunk sizes, schedule, dev probe) live in `config/pretrain.yaml`;
  optimizer peak LRs live in `config/optim.yaml`. `optim.encoder_lr_scale` deliberately does NOT
  apply to this stage.
* Watch `pretrain/acc` and `dev/acc`, not `pretrain/loss` -- random-projection targets have an
  irreducible entropy floor, so the loss curve flattens long before the representation does. Watch
  `train/branch_gain_max` for the amplitude escape (this stage starts from a fresh model, so its
  level is meaningful here in a way it is not in the warm-started transducer stage);
  `train/grad_norm` is ~99 % scalar gates, so
  `train/grad_norm_guarded` is the one that tracks model health.

```bash
# other knobs
python -m src.slices.PretrainEncoder.pretrain_bestrq --fresh --total-steps 240000
```

### Step 4: transducer training

Single joint stage: the 53.8 M-param Zipformer encoder, a `StatelessPredictor` and an additive
`TransducerJoiner`, trained together under
`rnnt_loss + ctc_aux_weight * ctc_loss + interctc_loss` (55.3 M params total). The aux CTC and
InterCTC taps are regularizers and health probes rather than separate stages.

#### The RNN-T objective: `full` (default) vs `pruned`

`training.transducer.rnnt_loss` selects between two implementations of the same loss.

* `full` (default) materialises the whole `[B,T,U+1,V]` joiner lattice. It is the reference the
  equivalence tests lock against, and it is what every published checkpoint was trained with.
* `pruned` evaluates the real joiner on an `s_range`-wide band of the alignment grid. A *linear*
  simple joiner (`simple_am_proj` + `simple_lm_proj`, no non-linearity, so its lattice is a
  transient inside the loss and never a stored activation) gives per-frame occupancy;
  `prune_ranges` turns that into band starts; the real joiner runs only on the band.

The two are locked together by test: at full band width the pruned recursion reproduces `rnnt_loss`
bit-for-bit in fp64, forward *and* gradient. On a real batch the pruned cost is a tight upper bound
on the full one (57.157 vs 57.076, +0.14 % at `s_range: 5`), and it can only ever be higher, because
pruning drops alignments.

**`pruned` is off because it measured 30 % slower, not because it failed.** Head to head on the same
card and the same allocator, 2026-08-02:

| objective | B | audio/step | ms/step | ×realtime | allocated | reserved | launches/step | mean kernel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` @ 28k frames | 20 | 274 s | 190.6 | **1438** | 7.41 GiB | 7.51 GiB | 20,166 | 17.1 µs |
| `pruned` @ 38k frames | 24 | 369 s | 334.9 | 1103 | 6.83 GiB | 6.96 GiB | 82,937 | 5.4 µs |

Pruning does save memory. It was adopted to lift a VRAM ceiling that turned out to be an artifact:
the full lattice is a large size-varying allocation, so under the *default* caching allocator it
reserves 9.71 GiB while allocating 5.25, and the trainer sets `expandable_segments:True`, under
which that gap is 0.10 GiB. With the ceiling gone, what remains is the launch count:
`rnnt_loss_pruned` scans frames in a Python loop through plain autograd, so backward retraces it,
and the step becomes launch-bound (5.4 µs mean kernel) where `full` is not (17.1 µs). Reach for
`pruned` when a lattice genuinely does not fit *unfragmented*. Before choosing it for speed,
give its scan a custom `autograd.Function` with an analytic fused backward, the same treatment
`RnntLoss._scan` already has (27.3 → 13.1 ms, bit-identical).

Under `pruned`, `prune_warmup_steps` (4000 batches) ramps the two terms from `(simple 1.0,
pruned 0.1)` to `(simple_loss_scale, 1.0)`: the bounds come from a freshly-initialised simple joiner,
so they are noise until it means something. Watch `train/rnnt_simple` in TensorBoard next to
`train/rnnt`: the simple loss is what picks the band, so it diverging corrupts the objective while
the pruned term still reads finite.

> **`pruned` adds two modules that `full` checkpoints do not contain.** Switching to it makes
> `transducer_avg.pt` and friends fail to load with `Missing key(s) in state_dict:
> simple_am_proj.weight, …`, which breaks Steps 5, 7, 8 and 9. The key is part of the architecture,
> not just the training loop.

```bash
python -m src.slices.TrainAcousticModel.train_transducer
#   re-running over a checkpoint from a different recipe? add --fresh to start from step 0
#   resuming a checkpoint written BEFORE model.trunk_norm? migrate it first -- it is missing the
#   10 trunk-normaliser parameters, and a plain resume fails on load_state_dict. Initialises each
#   log_scale from the measured per-stack trunk RMS and remaps AdamW's state by name:
PYTHONPATH=. python scripts/migrate_trunk_norm.py \
  --src data/checkpoints/transducer_step81000.pt --out data/checkpoints/transducer_last.pt
#   step too slow? where the time goes, per stage + per kernel (needs a free GPU, ~2 min):
PYTHONPATH=. python scripts/profile_transducer_step.py
#   compare the two RNN-T objectives on the same batches (overrides the config for the run):
PYTHONPATH=. python scripts/profile_transducer_step.py --rnnt-loss full   --steps 20
PYTHONPATH=. python scripts/profile_transducer_step.py --rnnt-loss pruned --steps 20
#   sweep the batch budget against peak VRAM without editing config between runs. Raise all THREE
#   budgets together or the one you left alone silently pins B (measured: with tokens held at 4000,
#   frames 32k/45k/58k all produced B=19 and identical 273 s audio/step):
PYTHONPATH=. python scripts/profile_transducer_step.py \
  --max-frames 60000 --max-tokens 7000 --max-lattice 2.6e7
#   how much of the lattice is U padding, and what token_sort_window would recover. Manifest only,
#   so it needs no GPU and takes seconds. Re-run it whenever the budgets move:
PYTHONPATH=. python scripts/measure_lattice_waste.py
```

* Trains on `data/manifests/train_sp2.jsonl` + the `train_sp2` mel cache by default, 562,482 rows,
  1,933 h. `--train-manifest data/manifests/train.jsonl --train-cache-split train` picks the clean
  961 h corpus; the two flags must move together or the fingerprint check rejects the pair.
* Warm-starts the encoder from `bestrq_encoder.pt`. Predictor, joiner and heads train from scratch.
* `total_steps` counts **loader batches, not optimizer updates**. At `grad_accum: 3` the shipped
  600k-batch run is 200,000 updates and 46,667 h = 24.1 passes over `train_sp2`. Convert to
  audio-hours (`total_steps × max_frames_per_batch / 100 / 3600`) before comparing any run against a
  published recipe. v1.0's `175000` read as a large number and was 9.1 passes over 961 h, ~6 % of
  the icefall reference, the dominant cause of its `test-other` gap, and going to 24.1 passes took
  greedy `test-other` from 11.44 % to 7.88 % with no architecture change.
* `token_sort_window: 1` (off) is the cheapest untried speed experiment here. On at 256/1024 it cuts
  `U` padding from 22.9 % of lattice cells to 5.6 %/2.0 % and epoch lattice work by ~18/21 %, worth
  roughly 6 % of wall time at the profiled split. It also raises the batch count ~2.6 %, so compare
  runs at matched **steps**, and it changes what a batch looks like (homogeneous speaking rate), so
  check dev ctc-WER before keeping it. Inert under `rnnt_loss: pruned`, and inert at any window near
  the batch size (~22).
* Budget derivation, both computed and the minimum taken: `audio_budget_steps = 144000 × 3600 / 274`
  = 1,891,971 (reference parity, 50 epochs × 2880 h); `wall_budget_steps = 50 × 3600 / 0.1906` =
  944,386 (50 h at the measured ms/step). Wall binds; rounded down to 900,000 to leave room for the
  ~90 full-dev validations and checkpointing. **Cut to 600,000 at step 365k** (2026-08-05): the WSD
  stable phase had gone flat to within noise (dev ctc-WER −0.0005 over 80k steps, per-eval σ ≈
  .0011), so the remaining 310k stable steps bought an unmeasurable amount while the anneal, worth
  −15 to −30 % relative, was still 25 h of compute away. Lowering `total_steps` under WSD only
  moves the decay window (675k → 450k); it does not re-heat the LR the way `cosine` would. Expect
  ~32 h of steps total (the run measured ~27 h at 0.163 s/step). **Before the anneal starts, copy
  `transducer_step448200.pt` out of the rotation**: `keep_last_n 5 × ckpt_every 5400` spans 27k
  steps, so a 150k-step anneal deletes the only pre-anneal resume point and with it the option to go
  back to a longer stable phase.
* **Cut again to 525,000 at step 419k, then reverted to 600,000 the same day**. The runaway that
  forced the second cut was not real (next bullet), so `total_steps` stands on the 365k reasoning
  above and nothing else. The cut is recorded because the abandoned run's history reads as if a
  divergence happened. It did not. Original text follows.
* **Cut again to 525,000 at step 419k** (2026-08-05), this time forced. The run went into a runaway:
  `train/grad_norm` median flat at 1.05 to 1.18 for 200k steps, then 1.30 → 1.79 → 11.0 over the
  360k → 400k window (p95 322), doubling every ~2.4k steps, while `encoder_param_norm`'s growth rate
  reversed a 200k-step deceleration (0.455 → 0.264 per 1k) into +2.728 per 1k. dev ctc-WER 0.0783 at
  400k (run best) → 0.0842 at 410k. Recovery is a rollback to `transducer_step394200.pt` plus
  `total_steps 525000`, whose `decay_start` = 393,750 puts the resume 450 steps into the anneal.
  **`grad_clip` cannot catch this**: Newton-Schulz renormalises Muon's update, so `clip_grad_norm_`
  is structurally inert for the 135 Muon matrices and the encoder takes a full-magnitude spectral
  step no matter how large the gradient is. Lowering the LR is the only live mid-run lever;
  `weight_decay` 1e-2 → 5e-2 in `config/optim.yaml` is the escalation.
* **That runaway was not a runaway** (2026-08-05, read from the checkpoints' optimizer moments).
  `train/grad_norm` is ~95 % the gradient of one scalar, `encoder.stacks.3.bypass`, which
  multiplies a whole `[B,T,C]` activation, so its gradient is a reduction over millions of terms and
  is sized by the batch shape rather than by anything about the model. Per-tensor AdamW
  `sqrt(Σ exp_avg_sq)`:

  | checkpoint | Muon (135 matrices) | AdamW total | `stacks.3.bypass` |
  |---|---|---|---|
  | `step394200` (healthy resume) | 0.7315 | 2.429 | 2.063 |
  | `step410400` (second abort) | 0.7535 | 3.287 | 3.023 |
  | `400k_predivergence` (median 11.0) | 0.7241 | 2.981 | 2.766 |
  | `step415800` (deep in the first abort) | **0.1743** | 4.994 | 4.987 |

  The encoder's own gradient scale is flat across every point called a runaway. The `GradNormGuard`
  aborted twice on it; the second abort (step 415,500) fired while dev ctc-WER was setting the run's
  record (0.0733 at 400k → **0.0680** at 410k) and the loss was falling. The guard now reads
  `train/grad_norm_guarded` (weight matrices, `ndim >= 2`); **watch that, not `train/grad_norm`.**
  What the checkpoints do show is the encoder gradient *collapsing* 4× (0.72 → 0.17) between 400k
  and 415.8k as dev WER regressed 0.0783 → 0.0842. That is the real failure signature, and one this
  one-sided guard does not catch.
* **The real divergence: one stack amplified until it silenced the next** (2026-08-09, from
  `runs/transducer` and the step286200 to step311326 checkpoints). Every branch inside a Zipformer
  block is *pre*-normed, so it contributes an O(1) correction however large the residual it is added
  to is, which means the amplitude a stack emits sets how much work every stack after it can do.
  Measured on a fresh block, 1−cos(input, output) against input RMS: 0.060 at 1.00, 0.023 at 1.65,
  0.009 at 2.72, 0.0004 at 12.18. The work falls off as 1/RMS².

  | step | 298k | 300k | 302k | 304k | 306k | 308k | 310k |
  |---|---|---|---|---|---|---|---|
  | stack 1 processed `b*g` | 4.52 | 4.83 | 5.07 | 5.98 | 8.80 | **12.18** | **12.18** |
  | stack 1 residual `1-b` | 0.25 | 0.26 | 0.28 | 0.27 | 0.16 | 0.00 | 0.00 |
  | stack 2 processed `b*g` | 7.00 | 7.19 | 7.18 | 7.04 | 5.53 | 2.58 | 1.18 |
  | stack 2 residual `1-b` | 0.43 | 0.41 | 0.41 | 0.42 | 0.55 | 0.79 | 0.90 |

  `train/grad_norm` 1.95 → **201** over the same window with `train/grad_norm_guarded` only
  0.82 → 1.60, i.e. ~99 % of the explosion was the scalar gates, whose gradient is proportional to
  the gain. **A one-sided cap at 2.5 only delayed this.** Full-dev greedy-CTC WER:

  | | 280k | 290k | 300k | 310k |
  |---|---|---|---|---|
  | uncapped gain | 0.0833 | 0.0848 | **0.1101** | *(aborted, rolled back to 275.5k)* |
  | capped at 2.5 (re-run) | 0.0835 | 0.0827 | 0.0818 | **0.1136** |

  The cap bought ~25k good steps and 0.0818 was the run's best, then the same collapse happened.
  `train/branch_gain_max` hit exp(2.5) = 12.18 at ~290k and stayed **pinned** there for the
  remaining 20k steps, binding continuously rather than clipping an excursion, while the pressure
  migrated: one `log_scale` at the ceiling in `step291600`, three in `step307800`.

  The driver is Adam meeting a sign-consistent gradient on a *scalar*: `|exp_avg|/√exp_avg_sq` is
  0.93/0.91/0.73 for the three runaway gains against a median of 0.14 over all 98. A ~0.9-coherent
  scalar marches at nearly the full LR every step no matter how small its gradient is, so nothing
  short of a bound stops it (weight decay's equilibrium at that coherence is |log_scale| ~ 93).
  `biasnorm_log_scale_max` is now **1.0**, so a stack can amplify by at most `e` and the next stack
  still does ~1/7th of its full-RMS work rather than 1/136th. The floor is **−2.0**, deliberately
  not symmetric: attenuation has no 1/RMS² argument against it and the healthy distribution runs
  low (p05 −1.54 over 97 encoder BiasNorms), so the earlier −1.0 clipped 15 of 97 gains in
  `bestrq_encoder.pt`, the normal population rather than a pathology.

  **The gain bound is not a bound on what a BiasNorm emits.** It divides by `rms(x − bias)` and
  scales `x`, so output RMS is `exp(log_scale) · rms(x)/rms(x − bias)`; the second factor is free.
  The 2026-08-21 run collapsed through it with `log_scale` pinned inside its bound
  (`stacks.1.blocks.1.norm_out` at 68.7× per frame, RMS 187 against an implied 2.72), which fed
  stack 2 an amplitude its own normalised branch could not match, so `bypass` went to exactly 0 and
  the stack deleted itself. `biasnorm_max_amplification: 4.0` floors the normaliser at `rms(x)/4`.

  The trainer warns on `train/gains_at_ceiling`, the *count* of gains resting on the ceiling,
  and only when it reaches a new high-water mark above what the run started with. A level test on
  `branch_gain_max` cannot work on a warm-started run: the pretrained encoder ships 6 gains already
  on the ceiling, so it warned on step 0 and on every log line after it. The count is what actually
  moved during the collapse (1 → 3 tensors). The baseline rides in the checkpoint's `extra`, so a
  resume keeps the run's own reference.
* **Bounded parameters are re-projected after every optimizer step.** `ZipformerStack.bypass` and
  `BiasNorm.log_scale` are both read through `clamp`, whose gradient is exactly 0 past the bound,
  so a parameter pushed out of range stopped training permanently: `encoder.stacks.5.bypass` sat at
  1.0020 to 1.0049 for the whole 394k to 416k window with AdamW `exp_avg` at −2.3e-22, and
  `stacks.3.bypass` climbed 0.9655 → 0.9872 toward the same trap. Both trainers now call
  `project_constraints(model)` (`shared_kernel/ParameterProjection.py`) after every step, so a
  parameter rests *on* its bound where gradient still flows. It walks the whole model, so it
  reaches the predictor's `BiasNorm` as well as the encoder's 98, and finds the stacks through the
  `_Checkpointed` wrapper.
* **Scalars are clipped per tensor, not against a shared norm.** Splitting the clip into
  (matrices | scalars) removed the coupling to Muon but only relocated it: read off
  `transducer_last.pt`, `encoder.stacks.3.bypass` sat at **4.9998** against a `grad_clip` of 5.0
  while the other five gates were at 0.007 to 0.025: one gate consuming the whole scalar clip budget
  every step, so all 414 other scalars (every bias, all 98 `log_scale`) were rescaled by a factor
  set by that one gate. Adam is scale-invariant to a *constant* rescale, not to one that swings
  ~100× step to step against EMAs that run across steps. `clip_grads_per_tensor` bounds each tensor
  against its own norm and still reports the group's pre-clip norm, so `train/grad_norm` stays
  comparable with the run history.
* **Weight decay applies only to `ndim >= 2`** (weight matrices and conv kernels). For every other
  parameter in this model zero is a specific degenerate setting, not a neutral shrink target, so
  decaying it is a standing pull away from whatever it learned: `bypass` → the stack skipped,
  `res_lambda` → value residual disabled (these learn −1.40…+0.47), `SimpleDownsample.weights` →
  pre-softmax logits forced to uniform pooling. `BiasNorm.log_scale` is exempted for a different
  reason. 0 there *is* the neutral value, but at these LRs decay is ~10× too weak to hold a
  sign-consistent gain (equilibrium |log_scale| ~ 93), so the projection does that job instead. 415
  tensors / 160,418 elements = 0.3 % of the model, so this is not a change in regularisation
  strength. Muon only ever holds 2-D matrices and is unaffected; AdamW splits 2 groups into 4.
* **Snapshots ahead of the resume point are deleted at startup.** Rotation is by step number, so a
  `transducer_step*.pt` from a run that was later rolled back outranked every snapshot the
  replacement run wrote and never aged out. After the rollback to 394200 a stale `step415800` sat
  in the directory, and `average_checkpoints.py --last-n 5` would have averaged that diverged model
  into the decode checkpoint.
* `warmup_steps` is `2500 × grad_accum`, i.e. 2,500 optimizer updates as v1.0 had. It protects the
  warm-started encoder from the fresh joiner's early gradient, a transient measured in updates, so
  it does **not** scale with `total_steps`.
* Checkpoints: `transducer_last.pt` (periodic resume point), `transducer_best.pt` (best full-dev
  greedy-CTC WER, the selection metric) and rolling `transducer_step{N}.pt` snapshots for Step 5.
* Rank runs on full-dev ctc-WER only. The v1.0 175k run ended at 0.0620; the 600k run reached
  **0.0408** at step 590k. The greedy-transducer probe is ~1k words and too noisy to compare across
  runs.
* **Do not judge a WSD run before its anneal.** On the 600k run dev ctc-WER moved 0.0833 → 0.0787
  across steps 250k to 450k, flat to within noise, then fell to 0.0408 over the `lr_decay_frac:
  0.25` window. Reading the stable phase as convergence would have discarded 48 % of the final
  quality.
* Telemetry: watch `dev/blank_frac` fall from ~1.000 and `dev/transducer_wer` in TensorBoard.

There are three batch budgets, not two, and **they must move together**, because each can
independently pin the batch, so the one left behind silently becomes the binding budget. Measured:
raising frames 32k → 45k → 58k with `max_tokens_per_batch` held at 4000 produced B = 18, 19, 19 and
an identical 273 s of audio per step. `max_frames_per_batch` (28000) and `max_tokens_per_batch`
(4400) are *sums*, so they set the average batch; `max_lattice_per_batch` (1.1e7, in units of `B ×
max(frames) × max(chars)`) caps the *worst* batch, which is what peak VRAM tracks. Without that cap
the densest batch of an epoch is 1.7× the p99.9 one, which is what produced periodic `CUDA OOM:
batch dropped` warnings.

The shipped budgets measure **B = 20, 274 s audio/step, 190.6 ms/step (1438× realtime), peak
7.41 GiB allocated / 7.51 GiB reserved**, of roughly 10.2 GiB this card leaves a training process
once the desktop has its share. The ~2.7 GiB left over is margin for the epoch's densest batch, and
the profiler's 20-step sample under-reads that batch, so do not spend it without re-profiling. On
OOM, lower `max_lattice_per_batch` first (it targets the tail directly, at almost no throughput
cost), then `max_frames_per_batch`, or set `training.transducer.grad_checkpoint: true` (~30 %
slower, bounds VRAM).

> **Never size a batch budget on peak VRAM read off the default allocator.** The RNN-T lattice is a
> large size-varying allocation; the default caching allocator rounds each one up to a fresh block,
> so reserved runs far above allocated (5.25 → 9.71 GiB) and reads as a ceiling that is not there.
> The trainer sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, under which the same lattice
> reserves 7.51 GiB against 7.41 allocated. `scripts/profile_transducer_step.py` sets it too, but
> anything else that measures VRAM must, or it is profiling a different allocator than the run.

One experiment is off by default, for these reasons:

* `cr_ctc` (`config/transducer.yaml`, default `false`) consistency-regularises the CTC head over two
  SpecAugment views (bidirectional KL, `cr_weight`) with RNN-T on view 1 only, and adds a second
  encoder forward. **It lost.** A 163.8k-step run of `cr_ctc: true` plus `--train-manifest
  data/manifests/train_sp.jsonl` (the *old* 3-way manifest, which the build script no longer emits)
  reached best dev transducer-WER 0.0674 against the shipped recipe's
  0.0553 (≈ +2 % test WER), with InterCTC and grad-norm climbing through the tail. That is
  over-regularisation on top of an aux stack already carrying `ctc_aux_weight` 0.2 and two InterCTC
  taps at 0.15. The run also moved the objective and the data recipe together, so **neither is
  isolated and the negative result is unattributed**. `_joint_loss_cr` also never applied the
  paper's 2.5× time-mask. Re-enabling means retuning `cr_weight`/`ctc_aux_weight`, pairing it with
  `grad_checkpoint: true` or a lower `max_frames_per_batch`, and changing one thing at a time.
  The `train/cr_ctc` TensorBoard scalar reads 0 while it is off.

#### Resuming and interrupting training

Every trainer shares one resumable harness: after each `save_every`/`ckpt_every` steps it atomically
writes `<name>_last.pt` (model, all optimizers, RNG, step and `resume_count`) via a temp file plus
`os.replace`, so an interrupted write never corrupts the live checkpoint.

* **Resume** by re-launching the same command. `resume=True` is the default, so training continues
  from `data/checkpoints/{bestrq,transducer,lm}_last.pt` with a fresh, non-repeating epoch seeded
  `base_seed + resume_count`.
* **Ctrl-C is safe.** SIGINT/SIGTERM are caught cooperatively: the loop finishes its current step,
  checkpoints, and exits cleanly.
* **Force a fresh run** with `--fresh`, which ignores `*_last.pt` and starts from step 0 on the
  transducer and LM trainers. You need it whenever the existing checkpoint came from a different
  recipe -- including any change to `num_codebooks`, which resizes `pred_head`.

```bash
python -m src.slices.TrainAcousticModel.train_transducer --fresh
```

> **Checkpoints written before the optimizer param-group merge cannot be resumed.** AdamW/Muon now
> bucket parameters by peak LR (2 groups each, instead of one group per parameter) so the fused
> multi-tensor kernels engage, and `Optimizer.load_state_dict` rejects the group-count change with
> `loaded state dict has a different number of parameter groups`. Model weights are unaffected and
> the update rule is identical, but the Adam moments are not portable, so start fresh.

### Step 5: checkpoint averaging (required)

```bash
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5
```

The trainer keeps a rolling window of `transducer_step{N}.pt` snapshots
(`training.transducer.keep_last_n`, default 5). This means the tail into
`data/checkpoints/transducer_avg.pt`, which `streaming_decode.py`, `evaluate.py` and `serve_demo.py`
all load by default. Run it before any of them.

Averaging is element-wise over the float tensors of the snapshots' state dicts, so the result is one
model of identical architecture and parameter count (not an ensemble): same VRAM, same RTF. It lands
nearer the centre of the basin the late iterates bounce around, which is worth real WER at zero
training cost. It also beats `transducer_best.pt`, whose lead over its neighbouring steps is partly
noise in the ~1k-word dev probe. The averaged payload is decode-only (`optimizers: []`, `step: 0`),
and resume still runs off `transducer_last.pt`.

### Step 6: STREAM-LM

Fetch the corpus, pack it to token bins, then train:

```bash
python scripts/download_lm_text.py && gunzip -k data/lm_text/librispeech-lm-norm.txt.gz

python - <<'PY'
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command
from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler

lm = get_config().lm
PrepareLmData_Handler(SentencePieceTokenizer("data/tokenizer/bpe500.model")).run(
    PrepareLmData_Command(
        "data/lm_text/librispeech-lm-norm.txt", "data/lm_data", lm.subset_words, lm.val_words
    )
)
PY

python -m src.slices.TrainLanguageModel.train_lm   # auto-resumes from lm_last.pt; --fresh to restart
tensorboard --logdir runs/lm
```

* A deep-narrow causal Transformer (GQA + QK-norm + value residual, tied embeddings; Muon + AdamW,
  warmup → cosine, bf16, z-loss) at d_model 512 / 16 layers (~44 M params) for 70,000 steps
  (~0.5 to 0.9 epoch of the ~803 M-word corpus). The earlier 320-d / 40k rescorer was capacity-bound
  at val ppl ~42, leaving oracle-floor headroom a stronger LM converts. Scoring is still one
  utterance-end forward per beam, so decode RTF is unaffected.
* Windows are document-masked: each position attends only its own corpus line, matching how a
  rescored hypothesis is scored at decode time. Validation perplexity is therefore *not* comparable
  to unmasked runs, because masked scoring is strictly harder and more honest.
* The shipped run reached val ppl 36.75 (best, step 56k of 70k). The curve is flat from ~35k onward
  inside a ±1.5 ppl noise band, so the 40k → 70k extension bought little; the win over the 320-d LM
  shows up in the decode table rather than the perplexity. `lm_best.pt` is a noise-minimum draw from
  that band, and the LM trainer keeps no step snapshots, so it cannot be checkpoint-averaged.
* Re-tune α and β on dev after any retrain (Step 8 does this automatically).

### Step 7: decode options

```bash
python -m src.slices.Decode.streaming_decode AUDIO.flac --offline   # full-context single pass
python -m src.slices.Decode.streaming_decode AUDIO.flac             # chunked streaming
```

* Flags: `--checkpoint` (default `transducer_avg.pt`), `--tokenizer`, `--offline`. Everything else
  (`chunk_size`, `beam_size`, `max_symbols`, `lm_weight`, `ilm_weight`, `length_bonus`,
  `lm_checkpoint`) comes from `config/decode.yaml`.
* The LM is off (`lm_weight: 0.0`) in the committed config until Step 6 has produced a checkpoint.
  The committed weights are the pure-acoustic regression lock.

### Step 8: evaluation

`--clean` and `--other` pick a LibriSpeech acoustic condition. Exactly one is required, with no
default, and each binds both manifests plus the report path: `--clean` scores `test-clean`, tunes on
`dev-clean`, writes `runs/eval/report-clean.json`; `--other` is `test-other` / `dev-other` /
`report-other.json`. Neither run can be reported against the other's weights or overwrite its file.

```bash
# Full table, both splits (each writes its own report)
python -m src.slices.Evaluate.evaluate --clean
python -m src.slices.Evaluate.evaluate --other

# Acoustic-only, no LM stage and no dev sweep
python -m src.slices.Evaluate.evaluate --clean --stages greedy_transducer,beam --no-tune

# Fixed weights, no dev sweep (explicit reproduction of a reported run)
python -m src.slices.Evaluate.evaluate --clean --lm-weight 0.5 --ilm-weight 0.3

# Liveness smoke, ~1 min. --limit/--tune-limit take an evenly STRIDED subsample, never a head
# slice: a manifest is sorted by uttid, so its head is a couple of speakers.
python -m src.slices.Evaluate.evaluate --clean --limit 30 --tune-limit 30 \
  --lm-grid 0.0,0.2,0.4 --ilm-grid 0.0,0.2 --lb-grid 0.0,0.5 --rtf-probe 10
```

Tuning is automatic unless `--no-tune` or an explicit `--lm-weight`/`--ilm-weight` is given. Dev is
decoded once per mode acoustic-only, then `--lm-grid` (α) × `--ilm-grid` (β, the ILME subtraction)
× `--lb-grid` (`length_bonus`, per-token) is swept over the cached scores for free. Offline and
streaming get their own triple, since streaming's weaker acoustic scores want a different balance.
Tuning also prints the n-best oracle WER, the floor any rescoring of that beam can reach. It never
touches test, so the headline WER stays an honest held-out metric.

`length_bonus` is swept because RNN-T acoustic scores are un-normalised sums, so a short hypothesis
can win on score while deleting words. The tuned value now reaches the live rescorer and is recorded
per mode in the report's `weights` block, so a report can no longer disagree with itself about which
bonus produced its numbers.

A 13 × 7 × 5 grid is 455 points per mode, so the sweep prints only its **top 10** rather than every
point. If any selected weight lands on the top of its own grid, it logs
`WARNING: … is the TOP of its grid`: that means the grid chose it, not the data. v1.0 pinned α at
0.6 in all four sweeps and nobody noticed, which is why the default grids are now α ≤ 1.2 and
β ≤ 0.6.

The run makes two passes. The *quality* pass runs all six stage × mode decodes concurrently in
worker processes (`--workers`, default `eval.workers` = 4) over the full manifest, because a single
decode leaves the GPU ~70 % idle waiting on Python (the beam is GIL-bound, so threads were worth
only 1.1×, while processes measured 1.8× at 85 % GPU) and WER does not care who else is on the card.
The *timing* pass re-runs the same configurations alone and serially over `--rtf-probe` (default
200) evenly strided utterances, because RTF and latency mean nothing under contention. They report
`null` if the probe is switched off. Offline latency is `null` by definition, since there are no
partials, and `finalize` is the post-encoder search and rescore cost, meaning what a live session
still owes after the audio stops.

Results for the shipped checkpoints, both splits, are in the [README](README.md#results). Note that
the RTF figures there come from the contention-free timing pass and supersede older ~0.12 / ~0.16
numbers measured with both passes sharing the GPU.

Tuned weights differ by condition and by mode. On the v1.5 checkpoints the sweep selected
α/β/`length_bonus` = 0.5/0.3/0.25 offline and 0.7/0.2/0.5 streaming on `--clean`, and 0.7/0.4/0.25
offline and 1.1/0.5/1.0 streaming on `--other`. Streaming wants more α and more of the prior
removed, because its acoustic scores are weaker.

Seven of those eight α/β values now land interior to the widened grids, which closes v1.0's open
question: its four sweeps all pinned α at the old 0.6 ceiling, so they were lower bounds rather than
optima. `length_bonus` has taken over the problem. Dev-other streaming selected 1.0, the top of its
five-point grid, and the run warned about it, so widen that axis before quoting the streaming-other
number as tuned.

### Step 9: local demo

```bash
python -m src.slices.Demo.serve_demo --lm-weight 0.5 --ilm-weight 0.3
```

`--lm-weight 0.5 --ilm-weight 0.3` is the dev-clean-tuned offline pair (streaming tuned to
α = 0.7, β = 0.2); drop
both flags for the faster acoustic-only decoder. `--beam-size`, `--checkpoint`, `--tokenizer`,
`--host` and `--port` are also available. Startup prints the resolved beam and LM settings, and the
page repeats them as a strip of chips (also served raw at `/config`), so a silent fallback to
acoustic-only is visible from the browser too. Transcripts are sentence-cased for display only.

The page takes a drag-and-dropped file, plays it back beside its transcript, and shows a live input
level while the microphone is open, and a flat meter is what separates a dead microphone from a
decode that heard nothing. The microphone needs a secure origin, so reach the demo at `127.0.0.1` or
`localhost`, not at a LAN address.

---

## 3. Tests

```bash
# Fast suite: 317 passed, 2 deselected
PYTHONPATH=. python -m pytest -q

# Slow GPU gates, individually
PYTHONPATH=. python -m pytest tests/slices/test_overfit_transducer.py -m slow -s  # loss drop >50 % on one batch
PYTHONPATH=. python -m pytest tests/slices/test_train_lm.py -m slow -s            # LM tiny overfit

# All slow gates
PYTHONPATH=. python -m pytest -m slow -s
```

## 4. Lint, format, types

Run after every change. Configuration lives in `pyproject.toml` and `.flake8`.

```bash
black src scripts tests       # format (--check to verify only)
flake8 src scripts tests      # style + unused imports, max line length 100
PYTHONPATH=. mypy src         # type check, expect 0 errors
```

## 5. Configuration reference

Every tunable is loaded and pydantic-validated by `get_config()`. There is no constants module, so
the YAML is authoritative.

| File | Parameters |
| --- | --- |
| `config/audio.yaml` | sample rate, n_mels, FFT/window/hop, CMVN epsilon |
| `config/augment.yaml` | SpecAugment masks (GPU batch op applied in `TransducerModel.joint_loss`) |
| `config/features.yaml` | log-mel cache directory and enable flag |
| `config/model.yaml` | encoder dims/layers/heads, conv kernel, dropout, RoPE base, `encoder_value_residual_lambda`, vocab size, and the amplitude bounds: `biasnorm_log_scale_min`/`_max`, `biasnorm_max_amplification`, `stack_bypass_min`, `stack_in_proj_max_sigma`, `trunk_norm`, `trunk_norm_log_scale_max` |
| `config/training.yaml` | `transducer`: objective (`rnnt_loss` = `pruned`\|`full`, `s_range`, `prune_warmup_steps`, `simple_loss_scale`), batch budgets (`max_frames_per_batch`, `max_tokens_per_batch`, `max_lattice_per_batch`), `grad_accum`, LR shape (`warmup_steps`, `total_steps`, `lr_schedule`, `lr_stable_ratio`, `lr_decay_frac`, `lr_min_ratio`), `chunk_sizes`, `warm_start`, `grad_checkpoint`, `compile_modules`, `token_sort_window`, `spec_augment`, `dev_wer_utts`, `keep_last_n` |
| `config/transducer.yaml` | `predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_aux_weight`, `interctc_layers`/`interctc_weights`, `cr_ctc`/`cr_weight` |
| `config/optim.yaml` | `optimizer` (`adamw`\|`muon+adamw`), `muon_lr`/`adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay`, `encoder_lr_scale` |
| `config/pretrain.yaml` | BEST-RQ: `lr_scale`, `codebook_size`/`codebook_dim`/`num_codebooks`, `mask_prob`/`mask_span`/`noise_std`, `stack_frames`, `chunk_sizes`, LR shape (`warmup_steps`, `total_steps`, `lr_schedule`, `lr_decay_frac`, `lr_min_ratio`), `grad_clip`/`log_every`/`save_every`, `dev_every`/`dev_batches`, `max_frames_per_batch`, `seed` |
| `config/lm.yaml` | STREAM-LM: `d_model`/`layers`/`heads`/`kv_groups`, `context_len`, `optimizer`/`muon_lr`/`lr_peak`/`z_loss`, schedule, `subset_words` |
| `config/decode.yaml` | `chunk_size`, `beam_size`, `max_symbols`, `lm_weight` (α), `ilm_weight` (β), `lm_checkpoint`, `length_bonus` |
| `config/eval.yaml` | `ablation_stages` (`greedy_transducer`/`beam`/`beam_lm`), `report_path` (`{split}` placeholder), `workers`, `rtf_probe_utts` |

```bash
# Validate pydantic loading without starting a training run
python -c "from src.shared_kernel.Config_Adapter import get_config; print(get_config().training.transducer)"
```

> **Critical dependencies.** Changing `vocab_size` in `model.yaml` invalidates the tokenizer, the
> CMVN statistics and every existing checkpoint, so retrain the tokenizer (Step 2) and recompute
> CMVN before resuming. Changing `lm.d_model` invalidates `lm_best.pt`, which then fails to load on
> a size mismatch. `training.transducer.rnnt_loss` decides whether the two simple projections exist
> at all, so it is part of the transducer's architecture, not just its training loop: a checkpoint
> saved under one value will not load under the other. It is read by every consumer of a transducer
> checkpoint, not only the trainer, so Steps 5, 7, 8 and 9 are affected too.

## 6. Smoke checks

Fast wiring checks for runtime validation, not genuine training.

```bash
# 3-step transducer run on dev (random init, no warm-start). --train-cache-split MUST move with
# --train-manifest: the default split is train_sp2, and its fingerprint will not match dev.jsonl.
python -m src.slices.TrainAcousticModel.train_transducer \
  --train-manifest data/manifests/dev.jsonl --train-cache-split dev \
  --dev-manifest data/manifests/dev.jsonl --dev-cache-split dev \
  --total-steps 3 --warm-start '' --fresh --log-dir runs/_smoke --ckpt-dir data/_smoke_ckpt
```
