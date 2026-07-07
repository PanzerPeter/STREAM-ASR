# STREAM

**S**peech **T**ranscription via **R**egularized **E**ncoder–**A**ttention **M**odeling.

A from-scratch, streaming-capable automatic speech recognition (ASR) system trained on
LibriSpeech `train-clean-100`. It targets a single RTX 5070 (12 GB, Blackwell sm_120) and is
built to stay hardware-efficient: state-of-the-art building blocks reimplemented in pure
PyTorch, with no k2/icefall dependency and no pretrained weights.

## Approach

A two-pass hybrid CTC/attention system built around a Zipformer acoustic encoder:

| Module | Component | Role |
|---|---|---|
| Acoustic encoder | Zipformer | log-mel → conv subsample → multi-rate Zipformer stacks with dynamic-chunk masking (~25 Hz output) |
| Decoder | CTC head + attention decoder | CTC drives the streaming first pass; a bidirectional attention decoder rescores the second pass |
| Language model | Causal Transformer LM | shallow fusion in the first-pass beam, plus n-best rescoring in the second |

A single set of encoder weights runs either offline (full context, best WER) or streaming
(small chunk, low latency) through dynamic-chunk masking. The full design rationale lives in
[`docs/superpowers/specs/2026-07-03-streaming-asr-zipformer-design.md`](docs/superpowers/specs/2026-07-03-streaming-asr-zipformer-design.md).

**Target for 100 h from scratch:** offline WER ≈ 6–8 %, streaming ≈ 8–10 %, RTF < 0.3. This
is not Whisper-scale — the goal is the best result achievable from this data on this GPU.

## Status

| Plan | Scope | State |
|---|---|---|
| 1 | Data foundation: environment, manifests, tokenizer, features, dataloader | **Done** |
| 2 | Zipformer encoder + CTC head + Stage-A training | **Done** — trained, dev WER **0.111** (greedy CTC) |
| 3 | Causal encoder (Phase 0) + attention decoder + joint CTC/attn + Stage-B training (Phase 1) + streaming/offline decode (Phase 2) | **Done** — design + implementation ([spec](docs/superpowers/specs/2026-07-05-stage-b-hybrid-ctc-attention-streaming-design.md)); awaits Stage-B retrain for streaming-capable checkpoint |
| 4 | Neural LM + two-pass LM fusion/rescore + `test-clean` evaluation | Planned |

## Layout

The repository follows a vertical-slice layout adapted from `VSA.md` for a training pipeline.
Slices communicate only through artifact files (manifests, tokenizer, checkpoints) and typed
dataclass DTOs, never by importing each other's internals.

```
config/                   # audio/augment/model/training tunables as YAML (pydantic-validated)
src/
  shared_kernel/          # pure transforms/adapters shared across slices
    Config_Adapter.py       # loads + validates config/*.yaml -> get_config()
    AudioIO_Adapter.py      # FLAC load (soundfile) + resample
    LogMel_Transform.py     # 80-bin log-mel frontend
    Tokenizer_Adapter.py    # SentencePiece BPE-500 wrapper
  slices/
    BuildManifest/          # LibriSpeech split → manifest.jsonl; train BPE tokenizer
    ComputeCmvn/            # global mean/var over train → data/features/cmvn.pt
    ExtractFeatures/        # SpecAugment + speed-perturb dataset/collator/sampler
    TrainAcousticModel/     # Zipformer encoder + CTC head + trainer (Plan 3: + attention decoder, streaming)
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

All tunables are read from `config/*.yaml` via `get_config()` — edit the YAML, no code change
required.

## Streaming / Offline Decode

The encoder is now causal-in-time (Phase 0): causal `Conv2dSubsampling` frontend, per-frame
`ConvModule` with `BiasNorm`, and RoPE `pos_offset`. This enables streaming inference through
`ZipformerEncoder.streaming_forward()` with exact equivalence to full-context batched `forward()`.

```bash
# Offline two-pass: full context, best WER ~8–10 %
PYTHONPATH=. .venv/bin/python -m src.slices.Decode.streaming_decode data/Val/dev-clean-0001.flac --offline

# Streaming two-pass: chunked inference, low latency ~10–12 %
PYTHONPATH=. .venv/bin/python -m src.slices.Decode.streaming_decode data/Val/dev-clean-0001.flac
```

Both modes use `data/checkpoints/stage_b_best.pt` as the default checkpoint and
`data/tokenizer/bpe500.model` for tokenization. Decoding parameters (chunk size, beam width,
left-context trim, language model weight) live in `config/decode.yaml`.

> ⚠ Stage-B requires a retrain with the causal encoder to produce a streaming-capable
> `stage_b_best.pt`. WER targets (8–10 % offline, 10–12 % streaming) are goals; measured results
> await the retrain + GPU eval run (user-owned).

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Notes

- Audio decode uses `soundfile`, not `torchaudio.load`: torchaudio 2.11 removed its native
  decode/metadata backends and now routes through TorchCodec, which requires FFmpeg. torchaudio
  is kept only for pure-tensor ops (resample, mel spectrogram).
- Speed perturbation follows the Kaldi 3-way convention: a factor of 0.9 slows and lengthens
  the audio, while 1.1 speeds it up and shortens it.

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 PanzerPeter.
