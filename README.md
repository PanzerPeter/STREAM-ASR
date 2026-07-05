# STREAM

**S**peech **T**ranscription via **R**egularized **E**ncoder–**A**ttention **M**odeling.

A from-scratch, streaming-capable automatic speech recognition (ASR) system trained on
LibriSpeech `train-clean-100`. The design targets a single RTX 5070 (12 GB, Blackwell
sm_120) and stays deliberately hardware-efficient: SOTA building blocks reimplemented in
pure PyTorch, no k2/icefall, no pretrained weights.

This is a speech-to-text project only. Despite the folder name, there is no summarization.

## Approach

A two-pass hybrid CTC/attention system with a Zipformer acoustic encoder:

| Module | Component | Role |
|---|---|---|
| Acoustic encoder | Zipformer | log-mel → conv subsample → multi-rate Zipformer stacks with dynamic-chunk masking (~25 Hz output) |
| Decoder | CTC head + attention decoder | CTC is the streaming first pass; a bidirectional attention decoder rescores the second pass |
| Language model | Causal Transformer LM | shallow fusion in the first-pass beam + n-best rescoring in the second |

The same encoder weights run offline (full context, best WER) or streaming (small chunk,
low latency) via dynamic-chunk masking. Full design rationale lives in
`docs/superpowers/specs/2026-07-03-streaming-asr-zipformer-design.md`.

**Honest target for 100 h from scratch:** offline WER ≈ 6–8 %, streaming ≈ 8–10 %,
RTF < 0.3. Not Whisper-scale — the goal is the best achievable from this data on this GPU.

## Status

| Plan | Scope | State |
|---|---|---|
| 1 | Data foundation: env, manifests, tokenizer, features, dataloader | **Done** |
| 2 | Zipformer encoder + CTC head + Stage-A training | **Done** — trained, dev WER **0.111** (greedy CTC) |
| 3 | Attention decoder + joint CTC/attn + dynamic-chunk (Stage-B trained); streaming decode next | Phase 1 done ([spec](docs/superpowers/specs/2026-07-05-stage-b-hybrid-ctc-attention-streaming-design.md)) |
| 4 | Neural LM + two-pass LM fusion/rescore + `test-clean` evaluation | Planned |

## Layout

The repo follows a vertical-slice layout adapted from `VSA.md` for a training pipeline.
Slices communicate only through artifact files (manifests, tokenizer, checkpoints) and
typed dataclass DTOs — never by importing each other's internals.

```
config/                   # audio/augment/model/training tunables as YAML (pydantic-validated)
src/
  shared_kernel/          # pure transforms/adapters shared across slices
    Config_Adapter.py       # loads+validates config/*.yaml -> get_config()
    AudioIO_Adapter.py      # FLAC load (soundfile) + resample
    LogMel_Transform.py     # 80-bin log-mel frontend
    Tokenizer_Adapter.py    # SentencePiece BPE-500 wrapper
  slices/
    BuildManifest/          # LibriSpeech split → manifest.jsonl; train BPE tokenizer
    ComputeCmvn/            # global mean/var over train → data/features/cmvn.pt
    ExtractFeatures/        # SpecAugment + speed-perturb dataset/collator/sampler
    TrainAcousticModel/     # Zipformer encoder + CTC head + Stage-A trainer (Plan 3: + attention decoder, streaming)
scripts/verify_env.py     # asserts Blackwell + working torch
tests/                    # shape / round-trip / count sanity tests
data/                     # LibriSpeech splits, manifests, tokenizer, cmvn, checkpoints (gitignored)
```

## Setup

Requires `uv` and an RTX 5070-class GPU. Python 3.12 is provisioned by `uv` (the system
Python is 3.14, which has no PyTorch wheels yet).

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python scripts/verify_env.py   # expect: OK: ... cap=(12, 0)
```

## Building the data foundation

```bash
# 1. Manifests (28539 / 2703 / 2620 utterances)
.venv/bin/python -c "from src.slices.BuildManifest.BuildManifest_Handler import build_manifest as b; \
from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand as C; \
[print(b(C(s, o))) for s, o in [ \
  ('data/Train/train-clean-100','data/manifests/train.jsonl'), \
  ('data/Val/dev-clean','data/manifests/dev.jsonl'), \
  ('data/Test/test-clean','data/manifests/test.jsonl')]]"

# 2. BPE-500 tokenizer
.venv/bin/python -c "from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer as t; \
from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand as C; \
print(t(C('data/manifests/train.jsonl','data/tokenizer/bpe500',500)))"

# 3. Global CMVN (80-bin mean/std over train) → data/features/cmvn.pt
PYTHONPATH=. .venv/bin/python scripts/compute_cmvn.py
```

## Training

```bash
# Stage A — Zipformer + CTC (done: 120k steps, dev WER 0.111 → data/checkpoints/stage_a_last.pt)
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_a

# Stage B — hybrid CTC/attention (U2++ bidirectional decoder + joint loss + dynamic-chunk).
# Warm-starts the encoder + CTC head from stage_a_last.pt → data/checkpoints/stage_b_last.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_b

# Monitor either run (separate terminal): loss / lr / dev WER / dev blank_frac
.venv/bin/tensorboard --logdir runs/stage_b
```

All tunables are read from `config/*.yaml` via `get_config()` — edit the YAML, no code change.
Streaming decode (Phase 2: stateful chunked inference + two-pass CTC→attention rescore) is
designed but not yet built; see the Stage-B design spec under `docs/superpowers/specs/`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Notes

- Audio decode uses `soundfile`, not `torchaudio.load` — torchaudio 2.11 removed its native
  decode/metadata backends and routes through TorchCodec (which needs FFmpeg). torchaudio is
  kept only for pure-tensor ops (resample, mel spectrogram).
- Speed perturbation follows the Kaldi 3-way convention: factor 0.9 slows/lengthens audio,
  1.1 speeds/shortens it.

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 PanzerPeter.
