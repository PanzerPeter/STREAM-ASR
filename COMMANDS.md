# STREAM ASR — Command Reference

## 1. End-to-End Pipeline (Quick Run)
Run these commands in sequence to execute the full pipeline from environment setup to local demo.

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
PYTHONPATH=. python -m src.slices.PretrainEncoder.pretrain_bestrq
#   Continue an interrupted pretrain: re-run the SAME command — it auto-resumes from
#   data/checkpoints/bestrq_last.pt (full state: model + optimizers + step + RNG) to total_steps (180k).

# Step 3: Train Stage-A (Zipformer + CTC)
python -m src.slices.TrainAcousticModel.train_stage_a
#   …or warm-start the encoder from the optional BEST-RQ pretrain above (SP4):
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; run_stage_a(d.replace(StageATrainCommand(), encoder_init='data/checkpoints/bestrq_encoder.pt'))"

# Step 4: Train Stage-B (Hybrid CTC/Attention, U2++)
python -m src.slices.TrainAcousticModel.train_stage_b

# Resume after an interrupt/crash (SP2): re-run the SAME train command — every trainer auto-resumes
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; run_stage_a(d.replace(StageATrainCommand(), resume=False))"

# Step 5: Streaming Decode (Phase 2 Verification)
PYTHONPATH=. python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac --offline
PYTHONPATH=. python -m src.slices.Decode.streaming_decode data/Val/dev-clean/1272/128104/1272-128104-0000.flac

# Step 6: Train Neural Language Model (STREAM-LM)
python scripts/download_lm_text.py && gunzip -k data/lm_text/librispeech-lm-norm.txt.gz
python -c "from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler as H; from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command as C; from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer as T; from src.shared_kernel.Config_Adapter import get_config as g; lm=g().lm; H(T('data/tokenizer/bpe500.model')).run(C('data/lm_text/librispeech-lm-norm.txt', 'data/lm_data', lm.subset_words, lm.val_words))"
python -m src.slices.TrainLanguageModel.train_lm

# Step 7: Evaluation (Acoustic + LM Grid Sweep)
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl --tune data/manifests/dev.jsonl

# Step 8: Launch Local Web UI Demo
PYTHONPATH=. python -m src.slices.Demo.serve_demo --lm-weight 0.3

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

# 2. Retrain BPE-500 on 960h transcripts (overwrites data/tokenizer/bpe500.model; invalidates old LM)
PYTHONPATH=. python scripts/train_tokenizer.py

# 3. CMVN over a 15% random sample of 960h (stats converge well before the full set)
PYTHONPATH=. python scripts/compute_cmvn.py

# 4. Precompute the fp16 log-mel cache for every split (~55 GB, one-time, CPU-heavy)
PYTHONPATH=. python scripts/precompute_features.py
```

* The cache (`data/features/mel/<split>.{f16,index.npy,header.json}`) is streamed via mmap so the
  training epoch loop is GPU-bound (no per-epoch FLAC decode / FFT).
* Retraining the tokenizer on 960h invalidates the old LM data + checkpoint — regenerate the LM
  (Step 6) after this build.
* SpecAugment is now a GPU batch op (`ExtractFeatures/SpecAugmentBatch.py`, SP1) — built but not yet
  wired into the trainers (that wiring is an SP5 task); speed-perturb was dropped.

### Step 2b: (Optional) BEST-RQ Encoder Pretrain (SP4)

Self-supervised masked-prediction pretrain of the Zipformer encoder on the 960h mel cache (labels
ignored), before any supervised Stage-A run. A frozen random-projection quantizer turns clean mel into
discrete targets; the encoder predicts them from span-masked input. Runs on the SP2 resumable harness
and the SP3 Muon+muP optimizer. **User-run** (long GPU job).

```bash
# Pretrain → data/checkpoints/bestrq_encoder.pt (encoder-only warm-start artifact)
#          + data/checkpoints/bestrq_last.pt   (full-state resume point)
PYTHONPATH=. python -m src.slices.PretrainEncoder.pretrain_bestrq
tensorboard --logdir runs/bestrq

# Then warm-start supervised Stage-A from the pretrained encoder (encoder_init field on the command):
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; run_stage_a(d.replace(StageATrainCommand(), encoder_init='data/checkpoints/bestrq_encoder.pt'))"
```

* **Resumable & SIGINT-safe (SP2):** like every trainer — see [Resuming & Interrupting Training](#resuming--interrupting-training-sp2) below.
* **Optimizer (SP3):** `config/optim.yaml` selects `muon+adamw` (2D hidden weights → Muon spectrally
  normalized updates; embeddings/biases/norms/heads → AdamW), with optional muP width-invariant LR
  scaling (`mup_enabled`, default off). Peak LRs (`muon_lr`, `adamw_lr`) are authoritative there.
* Pretrain knobs (codebook, mask, schedule, `grad_clip`/`log_every`/`save_every`) live in
  `config/pretrain.yaml`.

### Step 3: Train Stage-A (Zipformer + CTC)

* **Target:** ~120k steps, target dev WER 10-14%. Checkpoints save to `data/checkpoints/`.
* **Telemetry:** Watch `blank_frac` fall from ~1.000 via TensorBoard before WER begins moving.

```bash
tensorboard --logdir runs/stage_a

```

### Step 4: Train Stage-B (Hybrid CTC/Attention, U2++)

* Warm-starts from `stage_a_last.pt`. Uses joint loss ($0.3 \cdot \text{CTC} + 0.7 \cdot \text{attn}$) and dynamic-chunk masking $\{0, 16, 32\}$.
* **OOM Resolution:** Lower `stage_b.max_frames_per_batch` in `config/training.yaml`, or set `stage_b.grad_checkpoint: true` (~30% slower, bounds VRAM).

```bash
tensorboard --logdir runs/stage_b

```

### Resuming & Interrupting Training (SP2)

All three trainers (BEST-RQ pretrain, Stage-A, Stage-B) share the SP2 resumable harness: after every
`save_every` steps they atomically write `<name>_last.pt` (model + **all** optimizers + RNG + step +
`resume_count`) via a temp-file + `os.replace`, so an interrupted write never corrupts the live
checkpoint.

* **Resume** — just re-launch the **same** command. `resume=True` is the default, so training continues
  from `data/checkpoints/{bestrq,stage_a,stage_b}_last.pt` (fresh, non-repeating epoch seeded
  `base_seed + resume_count`).
* **Ctrl-C is safe** — SIGINT/SIGTERM are caught cooperatively: the loop finishes its current step,
  checkpoints, and exits cleanly (no mid-step corruption).
* **Force a fresh run** — pass `resume=False` to ignore any existing `_last.pt`:

```bash
# Fresh Stage-A (ignore stage_a_last.pt); swap the handler/command names for Stage-B or pretrain
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; run_stage_a(d.replace(StageATrainCommand(), resume=False))"
```

> ⚠️ **Old checkpoints are incompatible.** The SP2 schema stores an `"optimizers"` **list**; pre-SP2
> `*.pt` files (singular `"optimizer"`) raise `KeyError` on load — this hits decode/demo/eval and the LM
> too. Delete/regenerate `stage_a/b_*.pt` and `lm_*.pt` before loading them.

### Step 5: Streaming Decode Options

* **Offline (Two-pass):** Full context, target WER ~8-10%.
* **Streaming (Two-pass):** Chunked, cached, latency-optimized, target WER ~10-12%. LM fusion is disabled (`lm_weight: 0.0`) by default until Step 6.

### Step 6: Train Neural LM (STREAM-LM)

* Trains a deep-narrow causal Transformer (GQA + QK-norm + value-residual, tied embeddings; AdamW warmup $\rightarrow$ cosine, bf16).
* `total_steps=40000` $\approx$ 0.3-0.5 epoch over the ~803M-word corpus.

### Step 7: Evaluation Alternates

```bash
# Acoustic-only evaluation (LM off, α=0)
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl

# Evaluation with a fixed α (skip dev sweep, explicit reproducibility)
python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl --lm-weight 0.3

```

> ⚠️ **Note:** Tuning via `--tune` is executed exclusively on the dev set to keep the test headline WER an honest held-out metric.

---

## 3. Testing Suites

```bash
# Fast suite (Quick verification)
PYTHONPATH=. python -m pytest -q

# Component Overfitting & Smoke Gates (Slow)
PYTHONPATH=. python -m pytest tests/slices/test_overfit_one_batch.py -m slow -s  # Stage-A drop >50%
PYTHONPATH=. python -m pytest tests/slices/test_overfit_hybrid.py -m slow -s     # Stage-B drop >50%
PYTHONPATH=. python -m pytest tests/slices/test_stage_b_smoke.py -m slow -s      # 5 steps on dev
PYTHONPATH=. python -m pytest tests/slices/test_train_lm.py -m slow -s           # LM tiny overfit

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
| `config/augment.yaml` | SpecAugment masks (GPU batch op; full strength for Stage B). Speed-perturb dropped (SP1) |
| `config/model.yaml` | Encoder dims/layers/heads, conv kernel, dropout, RoPE base, `encoder_value_residual_lambda`, vocab size, attention-decoder dims/layers/heads, `decoder_value_residual_lambda` |
| `config/training.yaml` | `stage_a` + `stage_b` settings (`ctc_weight`, `reverse_weight`, `label_smoothing`, `chunk_sizes`, `warm_start`) |
| `config/decode.yaml` | `chunk_size`, `beam_size`, `rescore_lambda`, `rescore_ctc_weight`, `lm_weight` ($\alpha$), `lm_checkpoint` |
| `config/lm.yaml` | STREAM-LM: `d_model`/`layers`/`heads`/`kv_groups`, `context_len`, schedule, `subset_words` |
| `config/eval.yaml` | `ablation_stages`, `report_path` |
| `config/optim.yaml` | Optimizer stack (SP3): `optimizer` (`adamw`\|`muon+adamw`), `muon_lr`/`adamw_lr`, `muon_momentum`, `ns_steps`, `weight_decay`, `mup_enabled`/`mup_base_dims` |
| `config/pretrain.yaml` | BEST-RQ pretrain (SP4): `codebook_size`/`codebook_dim`, `mask_prob`/`mask_span`/`noise_std`, `stack_frames`, `warmup_steps`/`total_steps`, `grad_clip`/`log_every`/`save_every`, `seed` |

### Configuration Verification

```bash
# Validate Pydantic runtime loading & type checking without starting a full training run
python -c "from src.shared_kernel.Config_Adapter import get_config; print(get_config().training.stage_a)"

```

> ⚠️ **Critical Dependency:** Changing `vocab_size` in `model.yaml` invalidates your current tokenizer, CMVN matrices, and active checkpoints. If changed, you must retrain the tokenizer (Step 2) and recompute CMVN before resuming training.

---

## 6. Smoke & Debug Verification

*Fast wiring checks for runtime validation. Intended for short runs only, not genuine training.*

```bash
# 3-step Stage-A Smoke Run on Dev
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; print(run_stage_a(d.replace(StageATrainCommand(), train_manifest='data/manifests/dev.jsonl', total_steps=3, log_dir='runs/_smoke', ckpt_dir='data/_smoke_ckpt')))"

# 5-step Stage-B Smoke Run on Dev (Random Init Initialization)
python -c "from src.slices.TrainAcousticModel.StageBTrainer_Handler import run_stage_b; from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand; import dataclasses as d; print(run_stage_b(d.replace(StageBTrainCommand(), train_manifest='data/manifests/dev.jsonl', dev_manifest='data/manifests/dev.jsonl', total_steps=5, warm_start='', log_dir='runs/_smoke_b', ckpt_dir='data/_smoke_ckpt')))"
