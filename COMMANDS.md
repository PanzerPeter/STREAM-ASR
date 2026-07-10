# STREAM ASR — Command Reference

## 1. End-to-End Pipeline (Quick Run)
Run these commands in sequence to execute the full pipeline from environment setup to local demo.

```bash
# Step 1: Environment Setup
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python scripts/verify_env.py

# Step 2: Data Foundation (Manifests, Tokenizer, CMVN)
python -c "from src.slices.BuildManifest.BuildManifest_Handler import build_manifest as b; from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand as C; [print(b(C(s, o))) for s, o in [('data/Train/train-clean-100','data/manifests/train.jsonl'), ('data/Val/dev-clean','data/manifests/dev.jsonl'), ('data/Test/test-clean','data/manifests/test.jsonl')]]"
python -c "from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer as t; from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand as C; print(t(C('data/manifests/train.jsonl','data/tokenizer/bpe500',500)))"
PYTHONPATH=. python scripts/compute_cmvn.py

# Step 3: Train Stage-A (Zipformer + CTC)
python -m src.slices.TrainAcousticModel.train_stage_a

# Step 4: Train Stage-B (Hybrid CTC/Attention, U2++)
python -m src.slices.TrainAcousticModel.train_stage_b

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

* Builds train/dev/test jsonl manifests (28,539 / 2,703 / 2,620 utterances).
* Trains a **BPE-500** SentencePiece tokenizer to `data/tokenizer/bpe500.{model,vocab}`.
* Computes global 80-bin mean/std CMVN over train features into `data/features/cmvn.pt`.

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
| `config/augment.yaml` | Speed-perturb factors, SpecAugment masks (full strength for Stage B) |
| `config/model.yaml` | Encoder dims/layers/heads, conv kernel, dropout, RoPE base, vocab size, attention-decoder dims/layers/heads |
| `config/training.yaml` | `stage_a` + `stage_b` settings (`ctc_weight`, `reverse_weight`, `label_smoothing`, `chunk_sizes`, `warm_start`) |
| `config/decode.yaml` | `chunk_size`, `beam_size`, `rescore_lambda`, `lm_weight` ($\alpha$), `lm_checkpoint` |
| `config/lm.yaml` | STREAM-LM: `d_model`/`layers`/`heads`/`kv_groups`, `context_len`, schedule, `subset_words` |
| `config/eval.yaml` | `ablation_stages`, `report_path` |

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
