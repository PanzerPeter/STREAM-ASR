# COMMANDS — STREAM ASR

Every command assumes you are **at the repo root with the project venv auto-activated**, so
bare `python` already resolves to the 3.12 interpreter (`.venv/bin/python`) — no path prefix
or `cd` needed. (Outside an activated shell, prefix `.venv/bin/`; the system Python is 3.14
and has no PyTorch wheels.)

`-m module` form puts the repo root on `sys.path`, so it needs no `PYTHONPATH`. The
`scripts/*.py` and `pytest` forms do — that is module resolution, independent of venv
activation — hence the `PYTHONPATH=.` prefix on those.

---

## 1. Full Pipeline

Run in order: environment → data foundation → Stage-A → Stage-B training.

### Step 1: Environment

```bash
# Provision Python 3.12 + deps (RTX 5070 / CUDA 12.8 wheels via requirements extra-index)
uv venv .venv --python 3.12
uv pip install -r requirements.txt

# Gate: assert Blackwell sm_120 + working torch before any run (training is eager; torch.compile
# is avoided on this torch 2.11 + Blackwell build)
python scripts/verify_env.py          # expect: OK: ... cap=(12, 0)
```

### Step 2: Data Foundation

```bash
# Manifests: train-clean-100 / dev-clean / test-clean → 28539 / 2703 / 2620 utts
python -c "from src.slices.BuildManifest.BuildManifest_Handler import build_manifest as b; \
from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand as C; \
[print(b(C(s, o))) for s, o in [ \
  ('data/Train/train-clean-100','data/manifests/train.jsonl'), \
  ('data/Val/dev-clean','data/manifests/dev.jsonl'), \
  ('data/Test/test-clean','data/manifests/test.jsonl')]]"

# BPE-500 SentencePiece tokenizer → data/tokenizer/bpe500.{model,vocab}
python -c "from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer as t; \
from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand as C; \
print(t(C('data/manifests/train.jsonl','data/tokenizer/bpe500',500)))"

# CMVN (global 80-bin mean/std over train) → data/features/cmvn.pt
PYTHONPATH=. python scripts/compute_cmvn.py
```

### Step 3: Train Stage-A (Zipformer + CTC)

```bash
# Full ~120k-step run — reads all defaults from StageATrainCommand + config/*.yaml
python -m src.slices.TrainAcousticModel.train_stage_a

# Monitor (separate terminal): loss / lr / it_per_s / dev WER / dev blank_frac
tensorboard --logdir runs/stage_a
```

Terminal shows a startup config panel, periodic `step/total · loss · lr · it/s · eta`
lines, and `dev WER │ blank <frac>` on each validation (best-marked). `blank_frac` is the
share of frames whose argmax is the CTC blank; early CTC sits near `1.000` with WER pinned at
`1.0000` — watch `blank_frac` fall (and `--log-level DEBUG` sample decodes turn non-empty) as
alignment forms, which precedes any WER movement. Target greedy-CTC dev WER ≈ 10–14 %.
Checkpoints land in `data/checkpoints/`.

### Step 4: Train Stage-B (hybrid CTC/attention, U2++)

```bash
# Warm-starts encoder + CTC head from data/checkpoints/stage_a_last.pt, attaches a fresh
# bidirectional attention decoder, trains the joint loss (0.3·CTC + 0.7·attn) with dynamic-chunk
# masking sampled per batch from {0,16,32}. → data/checkpoints/stage_b_last.pt
python -m src.slices.TrainAcousticModel.train_stage_b

# Monitor (separate terminal): train/{loss,ctc,attn,lr} + dev WER (CTC-greedy) + dev blank_frac
tensorboard --logdir runs/stage_b
```

Same startup panel + step lines as Stage-A, with `loss (ctc <..> attn <..>) │ chunk <n>` per step.
Dev WER stays CTC-greedy in this stage; the two-pass attention-rescore / streaming decode is Phase 2
(the future `Decode` slice). If the run OOMs, lower `training.yaml` `stage_b.max_frames_per_batch`
(or set `stage_b.grad_checkpoint: true` for ~30 % slower but bounded VRAM).

---

## 2. Tests

```bash
# Fast suite (slow/GPU gates deselected) — expect 37 passed, 3 deselected
PYTHONPATH=. python -m pytest -q

# GPU correctness gates: single-batch overfit, loss must drop >50%
PYTHONPATH=. python -m pytest tests/slices/test_overfit_one_batch.py -m slow -s   # Stage-A (CTC)
PYTHONPATH=. python -m pytest tests/slices/test_overfit_hybrid.py -m slow -s      # Stage-B (joint)

# Stage-B trainer smoke (5-step run on dev: warm-start off, joint loss + chunk sampling + ckpt)
PYTHONPATH=. python -m pytest tests/slices/test_stage_b_smoke.py -m slow -s

# All slow gates
PYTHONPATH=. python -m pytest -m slow -s
```

---

## 3. Lint / Format / Types

Run after every change. Config: `pyproject.toml` (black, mypy) + `.flake8` (line 100).

```bash
black src scripts tests      # format (add --check to verify only)
flake8 src scripts tests     # style / unused imports (max line 100)
PYTHONPATH=. mypy src        # type check (0 errors)
```

---

## 4. Config

All tunables live in `config/*.yaml`, loaded and validated by pydantic via
`shared_kernel/Config_Adapter.get_config()` — the single authoritative source. Paths stay as
`StageATrainCommand` defaults. Derived values (`blank_id`, `logits_width`) are computed, not stored.

| File | Holds |
|---|---|
| `config/audio.yaml` | sample rate, n_mels, FFT/window/hop, CMVN eps |
| `config/augment.yaml` | speed-perturb factors, SpecAugment masks (restored to designed strength for Stage B) |
| `config/model.yaml` | encoder dims/layers/heads, conv kernel, dropout, RoPE base, vocab size; attention-decoder dims/layers/heads |
| `config/training.yaml` | `stage_a` + `stage_b` (Stage-B adds joint-loss weights `ctc_weight`/`reverse_weight`, `label_smoothing`, `chunk_sizes`, `warm_start`) |

To change a hyperparameter, edit the YAML — no code change:

```bash
# e.g. shorten a debug run: set training.yaml stage_a.total_steps, then
python -m src.slices.TrainAcousticModel.train_stage_a
# validate any edit loads + type-checks:
python -c "from src.shared_kernel.Config_Adapter import get_config; print(get_config().training.stage_a)"
```

> ⚠ Changing `model.yaml` `vocab_size` invalidates the existing tokenizer, CMVN, and checkpoints —
> retrain the tokenizer (§1 Step 2) and recompute CMVN before training.

---

## 5. Smoke / Debug

```bash
# 3-step Stage-A trainer smoke on dev (fast wiring check of the full loop + checkpoint save)
python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; \
from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; \
import dataclasses as d; \
print(run_stage_a(d.replace(StageATrainCommand(), train_manifest='data/manifests/dev.jsonl', \
  total_steps=3, log_dir='runs/_smoke', ckpt_dir='data/_smoke_ckpt')))"

# 5-step Stage-B smoke on dev (warm_start='' → random init; exercises joint loss + chunk sampling)
python -c "from src.slices.TrainAcousticModel.StageBTrainer_Handler import run_stage_b; \
from src.slices.TrainAcousticModel.StageBTrainer_Command import StageBTrainCommand; \
import dataclasses as d; \
print(run_stage_b(d.replace(StageBTrainCommand(), train_manifest='data/manifests/dev.jsonl', \
  dev_manifest='data/manifests/dev.jsonl', total_steps=5, warm_start='', \
  log_dir='runs/_smoke_b', ckpt_dir='data/_smoke_ckpt')))"
```
