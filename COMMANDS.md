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
PYTHONPATH=. python scripts/precompute_features.py           # fp16 log-mel cache (~55 GB, one-time)

# Step 3: self-supervised encoder pretrain
python -m src.slices.PretrainEncoder.pretrain_bestrq

# Step 4: transducer training (single joint stage, all defaults)
python -m src.slices.TrainAcousticModel.train_transducer
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
python -m src.slices.Demo.serve_demo --lm-weight 0.6 --ilm-weight 0.3
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
* The LM (Step 6) is tokenizer-specific, so regenerate its packed data and checkpoint whenever the
  tokenizer changes.

Speed perturbation is optional. `scripts/build_speed_perturb_manifest.py` writes `train_sp.jsonl`,
three copies of every train utterance at 0.9 / 1.0 / 1.1 with corrected `num_samples`. It is not
part of the default recipe: opt in via `--train-manifest`, and note that resampling per row forces
on-the-fly FLAC decode instead of reads from the clean mel cache.

### Step 3: BEST-RQ encoder pretrain

Self-supervised masked-prediction pretraining of the Zipformer encoder on the 960 h mel cache
(transcripts ignored). A frozen random-projection quantizer turns clean mel into discrete targets,
and the encoder predicts them from span-masked input. Long GPU job.

```bash
python -m src.slices.PretrainEncoder.pretrain_bestrq
tensorboard --logdir runs/bestrq
```

* Produces `data/checkpoints/bestrq_encoder.pt` (encoder-only warm-start artifact) and
  `bestrq_last.pt` (full-state resume point).
* The transducer trainer consumes the former by default (`training.transducer.warm_start`), so no
  extra flag is needed.
* Pretrain knobs (codebook, mask, schedule, `grad_clip`/`log_every`/`save_every`) live in
  `config/pretrain.yaml`; optimizer peak LRs live in `config/optim.yaml`.

### Step 4: transducer training

Single joint stage: the 53.8 M-param Zipformer encoder, a `StatelessPredictor` and an additive
`TransducerJoiner`, trained together under
`rnnt_loss + ctc_aux_weight * ctc_loss + interctc_loss` (55.3 M params total). The aux CTC and
InterCTC taps are regularizers and health probes rather than separate stages.

```bash
python -m src.slices.TrainAcousticModel.train_transducer
#   re-running over a checkpoint from a different recipe? add --fresh to start from step 0
#   step too slow? where the time goes, per stage + per kernel (needs a free GPU, ~2 min):
PYTHONPATH=. python scripts/profile_transducer_step.py
```

* Warm-starts the encoder from `bestrq_encoder.pt`. Predictor, joiner and heads train from scratch.
* Target 175k steps (`training.transducer.total_steps`) at ~225 ms/step on the 5070 (~770×
  realtime, ~11 h end to end). Measured, not projected.
* Checkpoints: `transducer_last.pt` (periodic resume point), `transducer_best.pt` (best full-dev
  greedy-CTC WER, the selection metric) and rolling `transducer_step{N}.pt` snapshots for Step 5.
* Rank runs on full-dev ctc-WER only. The shipped 175k run ended at 0.0620, with the
  greedy-transducer probe at 0.0444. That probe is ~1k words and too noisy to compare across runs.
* Telemetry: watch `dev/blank_frac` fall from ~1.000 and `dev/transducer_wer` in TensorBoard.

There are three batch budgets, not two. `max_frames_per_batch` (18000) and `max_tokens_per_batch`
(4000) are *sums*, so they set the average batch. `max_lattice_per_batch` (6.0e6, in units of
`B × max(frames) × max(chars)`) caps the *worst* batch, which is what peak VRAM tracks. Without it
the densest batch of an epoch is 1.7× the p99.9 one (9930 MiB reserved vs 8446 MiB with the cap),
which is what produced periodic `CUDA OOM: batch dropped` warnings. The cap costs 14 extra batches
in 60,347 and leaves mean batch size at 13.98.

Steady-state peak on the defaults is ~8.0 GiB allocated / ~8.6 GiB reserved of 12 GiB, and the
epoch's worst batch reaches ~8.4 GiB reserved. On OOM, lower `max_lattice_per_batch` first (it
targets the tail directly, at almost no throughput cost), then `max_frames_per_batch`, or set
`training.transducer.grad_checkpoint: true` (~30 % slower, bounds VRAM).

Two experiments are off by default, for these reasons:

* `token_sort_window` (default `1` = off) re-sorts by transcript length inside a sliding window of
  the duration sort, so a batch is homogeneous in `U` as well as `T`. Measured on `train.jsonl`:
  lattice padding waste 21.6 % → 13.9 % at an unchanged batch count, 138 → 126 ms/step, peak VRAM
  5.36 → 5.06 GiB. It stays off because making every batch homogeneous in transcript length is a
  training-semantics change no run has isolated, and the shipped recipe did not have it.
* `cr_ctc` (`config/transducer.yaml`, default `false`) consistency-regularises the CTC head over two
  SpecAugment views (bidirectional KL, `cr_weight`) with RNN-T on view 1 only, and adds a second
  encoder forward. **It lost.** A 163.8k-step run of `cr_ctc: true` plus `--train-manifest
  data/manifests/train_sp.jsonl` reached best dev transducer-WER 0.0674 against the shipped recipe's
  0.0553 (≈ +2 % test WER), with InterCTC and grad-norm climbing through the tail. That is
  over-regularisation on top of an aux stack already carrying `ctc_aux_weight` 0.2 and two InterCTC
  taps at 0.15. The run also moved the objective and the data recipe together, so neither is
  isolated. Re-enabling means retuning `cr_weight`/`ctc_aux_weight`, pairing it with
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
  recipe. BEST-RQ pretrain has no CLI, so pass `resume=False` on `BestRqPretrainCommand` or move the
  checkpoint aside.

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
python -m src.slices.Evaluate.evaluate --clean --lm-weight 0.6 --ilm-weight 0.2

# Liveness smoke, ~1 min. --limit/--tune-limit take an evenly STRIDED subsample, never a head
# slice: a manifest is sorted by uttid, so its head is a couple of speakers.
python -m src.slices.Evaluate.evaluate --clean --limit 30 --tune-limit 30 \
  --lm-grid 0.0,0.2,0.4 --ilm-grid 0.0,0.2 --rtf-probe 10
```

Tuning is automatic unless `--no-tune` or an explicit `--lm-weight`/`--ilm-weight` is given. Dev is
decoded once per mode acoustic-only, then `--lm-grid` (α) × `--ilm-grid` (β, the ILME subtraction)
is swept over the cached scores for free. Offline and streaming get their own pair, since
streaming's weaker acoustic scores want a different balance. Tuning also prints the n-best oracle
WER, the floor any rescoring of that beam can reach. It never touches test, so the headline WER
stays an honest held-out metric.

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

Tuned weights differ by condition: α = 0.6 with β = 0.2 offline / 0.3 streaming on `--clean`, and
α = 0.6 with β = 0.3 in both modes on `--other`. α pinned to 0.6, the top of the default
`--lm-grid`, in all four sweeps, so try `--lm-grid 0.0,0.2,0.4,0.6,0.8,1.0` before assuming it is
the optimum.

### Step 9: local demo

```bash
python -m src.slices.Demo.serve_demo --lm-weight 0.6 --ilm-weight 0.2
```

`--lm-weight 0.6 --ilm-weight 0.2` is the dev-tuned offline pair (streaming tuned to β = 0.3); drop
both flags for the faster acoustic-only decoder. `--beam-size`, `--checkpoint`, `--tokenizer`,
`--host` and `--port` are also available. Startup prints the resolved beam and LM settings, and the
page repeats them as a strip of chips (also served raw at `/config`), so a silent fallback to
acoustic-only is visible from the browser too. Transcripts are sentence-cased for display only.

The page takes a drag-and-dropped file, plays it back beside its transcript, and shows a live input
level while the microphone is open — a flat meter is what separates a dead microphone from a decode
that heard nothing. The microphone needs a secure origin, so reach the demo at `127.0.0.1` or
`localhost`, not at a LAN address.

---

## 3. Tests

```bash
# Fast suite: 213 passed, 2 deselected
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
| `config/model.yaml` | encoder dims/layers/heads, conv kernel, dropout, RoPE base, `encoder_value_residual_lambda`, vocab size |
| `config/training.yaml` | `transducer`: batch budgets (`max_frames_per_batch`, `max_tokens_per_batch`, `max_lattice_per_batch`, `token_sort_window`), `grad_accum`, LR shape (`warmup_steps`, `total_steps`, `lr_schedule`, `lr_stable_ratio`, `lr_decay_frac`, `lr_min_ratio`), `chunk_sizes`, `warm_start`, `grad_checkpoint`, `spec_augment`, `dev_wer_utts`, `keep_last_n` |
| `config/transducer.yaml` | `predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_aux_weight`, `interctc_layers`/`interctc_weights`, `cr_ctc`/`cr_weight` |
| `config/optim.yaml` | `optimizer` (`adamw`\|`muon+adamw`), `muon_lr`/`adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay`, `encoder_lr_scale` |
| `config/pretrain.yaml` | BEST-RQ: `codebook_size`/`codebook_dim`, `mask_prob`/`mask_span`/`noise_std`, `stack_frames`, `warmup_steps`/`total_steps`, `grad_clip`/`log_every`/`save_every`, `seed` |
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
> a size mismatch.

## 6. Smoke checks

Fast wiring checks for runtime validation, not genuine training.

```bash
# 3-step transducer run on dev (random init, no warm-start)
python -m src.slices.TrainAcousticModel.train_transducer \
  --train-manifest data/manifests/dev.jsonl --dev-manifest data/manifests/dev.jsonl \
  --total-steps 3 --warm-start '' --fresh --log-dir runs/_smoke --ckpt-dir data/_smoke_ckpt
```
