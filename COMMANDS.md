# STREAM ASR — Command Reference

## 1. End-to-End Pipeline (Quick Run)
Run these commands in sequence to execute the full pipeline from environment setup to local demo.

> **Conventions:** all commands assume you are `cd`'d at the repo root with the venv **activated**
> (`source .venv/bin/activate`), so `python` is the 3.12 venv interpreter. `python -m …` and
> `python -c …` already have the repo root on `sys.path`; only `scripts/*.py` that import `src`,
> `pytest`, and `mypy` need the `PYTHONPATH=.` prefix.

```bash
# Step 1: Environment Setup
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python scripts/verify_env.py

# Step 2: Data Foundation — 960h build
PYTHONPATH=. python scripts/build_manifests.py       # all 5 splits, parallel probe
PYTHONPATH=. python scripts/train_tokenizer.py        # retrain BPE-500 on 960h train
PYTHONPATH=. python scripts/compute_cmvn.py           # CMVN over a 15% sample
PYTHONPATH=. python scripts/precompute_features.py    # fp16 log-mel cache (~55 GB, one-time)

# Step 2b (optional): BEST-RQ self-supervised encoder pretrain (SP4) → data/checkpoints/bestrq_encoder.pt
python -m src.slices.PretrainEncoder.pretrain_bestrq
#   Continue an interrupted pretrain: re-run the SAME command — it auto-resumes from
#   data/checkpoints/bestrq_last.pt (full state: model + optimizers + step + RNG) to total_steps (180k).

# Step 3: Train the transducer (single joint stage; encoder + StatelessPredictor + TransducerJoiner,
# warm-started from data/checkpoints/bestrq_encoder.pt by default — training.transducer.warm_start)
python -m src.slices.TrainAcousticModel.train_transducer
tensorboard --logdir runs/transducer

# Step 4 (optional): average the tail of the rolling transducer_step*.pt snapshots
# (training.transducer.keep_last_n) into one decode checkpoint
python scripts/average_checkpoints.py --last-n 5   # -> data/checkpoints/transducer_avg.pt

# Start new transducer model training
python -c "from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer; from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand; import dataclasses as d; run_transducer(d.replace(TransducerTrainCommand(), resume=False))"

# Step 5: Streaming Decode (single-pass RNN-T beam search)
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac --offline
python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac

# Step 6: Train Neural Language Model (STREAM-LM)
python scripts/download_lm_text.py && gunzip -k data/lm_text/librispeech-lm-norm.txt.gz
python -c "from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler as H; from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command as C; from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer as T; from src.shared_kernel.Config_Adapter import get_config as g; lm=g().lm; H(T('data/tokenizer/bpe500.model')).run(C('data/lm_text/librispeech-lm-norm.txt', 'data/lm_data', lm.subset_words, lm.val_words))"
python -m src.slices.TrainLanguageModel.train_lm
tensorboard --logdir runs/lm


# Step 7: Evaluation (Acoustic beam + LM n-best rescoring)
# --tune decodes dev ONCE acoustic-only, sweeps lm_weight (alpha) x ilm_weight (beta, the ILME
# subtraction) over the cached (acoustic, LM-sequence, internal-LM-sequence) scores for free,
# freezes the best pair, then runs the test table (greedy/beam/beam_lm x offline/streaming) with it.
# It also prints the n-best oracle WER = the floor any rescoring of that beam can reach.
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl --tune data/manifests/dev.jsonl
# Quick liveness smoke (finishes in ~1-2 min): a tiny dev subset + coarse grid + capped test table.
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl --tune data/manifests/dev.jsonl --tune-limit 30 --lm-grid 0.0,0.2,0.4 --ilm-grid 0.0,0.2 --limit 30

# Step 8: Launch Local Web UI Demo. 0.6 / 0.2 = the (alpha, beta) tuned on dev, i.e. the setup the
# 3.55 % test-clean number was measured at; drop both flags for the faster acoustic-only decoder.
# Startup prints the resolved beam/LM settings. Transcripts are sentence-cased for display only.
python -m src.slices.Demo.serve_demo --lm-weight 0.6 --ilm-weight 0.2

```

---

## 2. Detailed Pipeline Steps & Options

### Step 1: Environment Setup

* **Verification Gate:** Expect `"OK: ... cap=(12, 0)"` (Blackwell sm_120).
* **Hardware Target:** RTX 5070 / CUDA 12.8 wheels via extra-index.

### Step 2: Data Foundation

* Builds 5-split jsonl manifests for the 960h set: train **281,241** / dev-clean 2,703 / dev-other
  2,864 / test-clean 2,620 / test-other 2,939 utterances.
* Trains a **BPE-500** SentencePiece tokenizer to `data/tokenizer/bpe500.{model,vocab}`.
* Computes global 80-bin mean/std CMVN over train features into `data/features/cmvn.pt`.

#### 960h data build (SP1)

Scale-out over the full `train-clean-100 + train-clean-360 + train-other-500` (960h) set plus
`dev-{clean,other}` / `test-{clean,other}`. Heavy passes are **user-run** (GPU/CPU-bound). Run in order:

```bash
# 1. Manifests for all 5 splits (parallel soundfile probe; seconds)
PYTHONPATH=. python scripts/build_manifests.py

# 2. Train BPE-500 on 960h transcripts (writes data/tokenizer/bpe500.model; the LM must match this tokenizer)
PYTHONPATH=. python scripts/train_tokenizer.py

# 3. CMVN over a 15% random sample of 960h (stats converge well before the full set)
PYTHONPATH=. python scripts/compute_cmvn.py

# 4. Precompute the fp16 log-mel cache for every split (~55 GB, one-time, CPU-heavy)
PYTHONPATH=. python scripts/precompute_features.py
```

* The cache (`data/features/mel/<split>.{f16,index.npy,header.json}`) is streamed via mmap so the
  training epoch loop is GPU-bound (no per-epoch FLAC decode / FFT).
* The LM (Step 6) is tokenizer-specific — regenerate its packed data + checkpoint whenever the
  tokenizer changes.
* SpecAugment runs as a GPU batch op (`ExtractFeatures/SpecAugmentBatch.py`), applied inside
  `TransducerModel.joint_loss` on the train path only — gated by `training.spec_augment` (default
  `true`).

### Step 2b: (Optional) BEST-RQ Encoder Pretrain (SP4)

Self-supervised masked-prediction pretrain of the Zipformer encoder on the 960h mel cache (labels
ignored), before the joint transducer run. A frozen random-projection quantizer turns clean mel into
discrete targets; the encoder predicts them from span-masked input. Runs on the SP2 resumable harness
and the SP3 Muon+muP optimizer. **User-run** (long GPU job).

```bash
# Pretrain → data/checkpoints/bestrq_encoder.pt (encoder-only warm-start artifact)
#          + data/checkpoints/bestrq_last.pt   (full-state resume point)
python -m src.slices.PretrainEncoder.pretrain_bestrq
tensorboard --logdir runs/bestrq

# The transducer trainer warm-starts from this checkpoint by default
# (training.transducer.warm_start = data/checkpoints/bestrq_encoder.pt) — no extra flag needed:
python -m src.slices.TrainAcousticModel.train_transducer
```

* **Resumable & SIGINT-safe (SP2):** like every trainer — see [Resuming & Interrupting Training](#resuming--interrupting-training-sp2) below.
* **Optimizer (SP3):** `config/optim.yaml` selects `muon+adamw` (2D hidden weights → Muon spectrally
  normalized updates; embeddings/biases/norms/heads → AdamW), with optional muP width-invariant LR
  scaling (`mup_enabled`, default off). Peak LRs (`muon_lr`, `adamw_lr`) are authoritative there.
* Pretrain knobs (codebook, mask, schedule, `grad_clip`/`log_every`/`save_every`) live in
  `config/pretrain.yaml`.

### Step 3: Train the transducer (single joint stage)

* Single-pass streaming RNN-T: the unchanged ~53.8M-param Zipformer encoder + a `StatelessPredictor`
  (icefall-style, no recurrence) + an additive `TransducerJoiner`, trained jointly with
  `rnnt_loss + ctc_aux_weight * ctc_loss + interctc_loss` (aux CTC + InterCTC taps at stacks 3–4 are
  regularizers/health probes, not separate stages). Model is ~55.3M params total.
* Warm-starts the encoder from `data/checkpoints/bestrq_encoder.pt` by default
  (`training.transducer.warm_start`); predictor/joiner/heads always train from scratch.
* **Target:** ~120k steps (`training.transducer.total_steps`). Checkpoints:
  `data/checkpoints/transducer_last.pt` (periodic) / `transducer_best.pt` (best dev
  greedy-transducer WER). Reference: the 2026-07-20 run hit best dev
  transducer-WER **0.0637**.
* **Telemetry:** Watch `dev/blank_frac` fall from ~1.000 and `dev/transducer_wer` via TensorBoard.
* **OOM resolution:** lower `training.transducer.max_frames_per_batch` (18000) or
  `max_tokens_per_batch` (4000 — bounds the `B*T*(U+1)` RNN-T joiner lattice), or set
  `training.transducer.grad_checkpoint: true` (~30% slower, bounds VRAM).

```bash
tensorboard --logdir runs/transducer

```

### Resuming & Interrupting Training (SP2)

Every trainer (BEST-RQ pretrain, transducer) shares the SP2 resumable harness: after every
`save_every`/`ckpt_every` steps it atomically writes `<name>_last.pt` (model + **all** optimizers +
RNG + step + `resume_count`) via a temp-file + `os.replace`, so an interrupted write never corrupts
the live checkpoint.

* **Resume** — just re-launch the **same** command. `resume=True` is the default, so training continues
  from `data/checkpoints/{bestrq,transducer}_last.pt` (fresh, non-repeating epoch seeded
  `base_seed + resume_count`).
* **Ctrl-C is safe** — SIGINT/SIGTERM are caught cooperatively: the loop finishes its current step,
  checkpoints, and exits cleanly (no mid-step corruption).
* **Force a fresh run** — pass `resume=False` to ignore any existing `_last.pt`:

```bash
# Fresh transducer run (ignore transducer_last.pt); swap the handler/command names for pretrain
python -c "from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer; from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand; import dataclasses as d; run_transducer(d.replace(TransducerTrainCommand(), resume=False))"
```

### Step 5: Streaming Decode Options

* **Offline:** Full-context single encoder pass, then one RNN-T beam search over the whole memory —
  test-clean WER 3.6–4.4% (beam+LM / beam).
* **Streaming:** Chunked/cached encoder (`StreamCache`) feeding the same beam search as frames
  arrive — test-clean WER 4.7–6.1%, ~27 ms first-partial latency. LM off (`lm_weight: 0.0`) by
  default until Step 6.
* `--offline` selects offline mode; omitting it runs streaming (the `streaming_decode.py` default).

### Step 6: Train Neural LM (STREAM-LM)

* Trains a deep-narrow causal Transformer (GQA + QK-norm + value-residual, tied embeddings; Muon+AdamW
  warmup $\rightarrow$ cosine, bf16, z-loss). Windows are **document-masked**: each position attends
  only its own corpus line, matching how a rescored hypothesis is scored at decode time.
* `total_steps=40000` $\approx$ 0.3-0.5 epoch over the ~803M-word corpus. At the tuned α=0.6, β=0.2
  the LM+ILME buys −0.86/−1.39 abs WER offline/streaming over acoustic-only. (Val ppl is not
  comparable to pre-masking runs — masked scoring is a strictly harder, and more honest, metric.)

### Step 7: Evaluation Alternates

```bash
# Acoustic-only evaluation (LM off, α=0)
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl

# Evaluation with fixed weights (skip dev sweep, explicit reproducibility; 0.6/0.2 = tuned optimum)
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl --lm-weight 0.6 --ilm-weight 0.2

# test-other (n=2,939): the hard split. α re-tuned on dev-other so the acoustic condition matches;
# --report is mandatory here, otherwise it overwrites the test-clean report.
python -m src.slices.Evaluate.evaluate data/manifests/test-other.jsonl \
  --tune data/manifests/dev-other.jsonl --report runs/eval/report-other.json

```

> ⚠️ **Note:** Tuning via `--tune` is executed exclusively on the dev set to keep the test headline WER an honest held-out metric.

**Reference results (test-clean, n=2,620, 2026-07-23 run, tuned α=0.6/β=0.2) → `runs/eval/report.json`:**

| Stage | Offline WER / CER | Streaming WER / CER |
| --- | --- | --- |
| `greedy_transducer` | 4.61% / 1.60% | 6.36% / 2.28% |
| `beam` | 4.41% / 1.52% | 6.07% / 2.13% |
| **`beam_lm`** | **3.55% / 1.24%** | **4.68% / 1.67%** |

RTF stays ≪ 1 across the board (offline ~0.12, streaming ~0.16 with LM, measured with both passes
sharing the GPU).

---

## 3. Testing Suites

```bash
# Fast suite (Quick verification)
PYTHONPATH=. python -m pytest -q

# Component Overfitting & Smoke Gates (Slow)
PYTHONPATH=. python -m pytest tests/slices/test_overfit_transducer.py -m slow -s  # transducer loss drop >50% on one batch
PYTHONPATH=. python -m pytest tests/slices/test_train_lm.py -m slow -s            # LM tiny overfit

# Run all slow gates
PYTHONPATH=. python -m pytest -m slow -s

```

---

## 4. Linting, Formatting & Types

Run after every change. Configurations are governed by `pyproject.toml` and `.flake8`.

```bash
black src scripts tests       # Format code (--check to verify only)
flake8 src scripts tests      # Check style and unused imports (max line length 100)
PYTHONPATH=. mypy src         # Type check checking (Strict: expect 0 errors)

```

---

## 5. Configuration Architecture

| File | Parameters Controlled |
| --- | --- |
| `config/audio.yaml` | Sample rate, n_mels, FFT/window/hop, CMVN epsilon |
| `config/augment.yaml` | SpecAugment masks (GPU batch op, applied in `TransducerModel.joint_loss`, gated by `training.spec_augment`). Speed-perturb dropped (SP1) |
| `config/model.yaml` | Encoder dims/layers/heads, conv kernel, dropout, RoPE base, `encoder_value_residual_lambda`, vocab size |
| `config/training.yaml` | `transducer` settings (SP5): `max_frames_per_batch`, `max_tokens_per_batch`, `grad_accum`, `warmup_steps`/`total_steps`, `chunk_sizes`, `warm_start`, `grad_checkpoint`, `dev_wer_utts`, `keep_last_n` |
| `config/transducer.yaml` | Transducer architecture (SP5): `predictor_dim`, `predictor_context`, `joiner_dim`, `ctc_aux_weight`, `interctc_layers`/`interctc_weights` |
| `config/decode.yaml` | `chunk_size`, `beam_size`, `max_symbols`, `lm_weight` ($\alpha$), `ilm_weight` ($\beta$, ILME subtraction), `lm_checkpoint`, `length_bonus` |
| `config/lm.yaml` | STREAM-LM: `d_model`/`layers`/`heads`/`kv_groups`, `context_len`, `optimizer`/`muon_lr`/`lr_peak`/`z_loss`, schedule, `subset_words` |
| `config/eval.yaml` | `ablation_stages` (`greedy_transducer`/`beam`/`beam_lm`), `report_path` |
| `config/optim.yaml` | Optimizer stack (SP3): `optimizer` (`adamw`\|`muon+adamw`), `muon_lr`/`adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay`, `mup_enabled`/`mup_base_dims` |
| `config/pretrain.yaml` | BEST-RQ pretrain (SP4): `codebook_size`/`codebook_dim`, `mask_prob`/`mask_span`/`noise_std`, `stack_frames`, `warmup_steps`/`total_steps`, `grad_clip`/`log_every`/`save_every`, `seed` |

### Configuration Verification

```bash
# Validate Pydantic runtime loading & type checking without starting a full training run
python -c "from src.shared_kernel.Config_Adapter import get_config; print(get_config().training.transducer)"

```

> ⚠️ **Critical Dependency:** Changing `vocab_size` in `model.yaml` invalidates your current tokenizer, CMVN matrices, and active checkpoints. If changed, you must retrain the tokenizer (Step 2) and recompute CMVN before resuming training.

---

## 6. Smoke & Debug Verification

*Fast wiring checks for runtime validation. Intended for short runs only, not genuine training.*

```bash
# 3-step transducer Smoke Run on Dev (random init, no BEST-RQ warm-start)
python -c "from src.slices.TrainAcousticModel.TransducerTrainer_Handler import run_transducer; from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand; import dataclasses as d; print(run_transducer(d.replace(TransducerTrainCommand(), train_manifest='data/manifests/dev.jsonl', dev_manifest='data/manifests/dev.jsonl', total_steps=3, warm_start='', log_dir='runs/_smoke', ckpt_dir='data/_smoke_ckpt')))"
