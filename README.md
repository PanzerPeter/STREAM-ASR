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
(small chunk, low latency) through dynamic-chunk masking.

**Value residual (ResFormer/SVFormer).** The Zipformer stacks, the attention decoder, and STREAM-LM
all use value-residual attention: block/layer 0 of each stack injects its attention *values* into the
deeper blocks, adding a gradient shortcut through attention that trains stably at depth so the same
accuracy is reachable at a narrower width. In the encoder and decoder the mix is a **learnable
per-block gate initialised to 0** (`encoder_value_residual_lambda` / `decoder_value_residual_lambda`
set the init): a fresh model therefore starts *identical* to a no-value-residual baseline and the
residual grows only as far as training wants it. This matters because Stage-A CTC sits on a
blank-collapse knife-edge — a fixed non-zero gate destabilises the escape, whereas the zero-init gate
does not. The encoder's streaming path caches post-residual values, so `streaming_forward` stays
exactly equal to the chunked `forward` (`test_streaming_forward_equivalence`).

**Target for 100 h from scratch:** offline WER ≈ 6–8 %, streaming ≈ 8–10 %, RTF < 0.3. This
is not Whisper-scale — the goal is the best result achievable from this data on this GPU.

## Status

| Plan | Scope | State |
|---|---|---|
| 1 | Data foundation: environment, manifests, tokenizer, features, dataloader | **Done** |
| 2 | Zipformer encoder + CTC head + Stage-A training | Code done; **retrain pending** — value residual added to the encoder |
| 3 | Causal encoder (Phase 0) + attention decoder + joint CTC/attn + Stage-B training (Phase 1) + streaming/offline decode (Phase 2) | Code done; **retrain pending** — value residual added to encoder + decoder |
| 4 | Neural LM (STREAM-LM) + two-pass LM fusion/rescore + Evaluate slice | **Done** — LM trained (val ppl **16.4**, unchanged); two-pass decode fixes + parallel eval landed |
| — | Local demo (web UI): upload a file or speak live | **Done** — `Demo` slice; verified end-to-end |

> **Retrain required.** Adding value residual changed the encoder and attention-decoder
> computation, so the Stage-A and Stage-B checkpoints were removed and must be regenerated (STREAM-LM
> is unchanged and kept). Prior WER figures (Stage-A dev 0.111, Stage-B dev 0.0999, test-clean
> 9.1 %/12.0 %) were measured **before** value residual and are superseded; new numbers land after the
> retrain below.

## Layout

The repository follows a vertical-slice layout adapted from `VSA.md` for a training pipeline.
Slices communicate only through artifact files (manifests, tokenizer, checkpoints) and typed
dataclass DTOs, never by importing each other's internals.

```
config/                   # audio/augment/model/training/decode/lm/eval tunables as YAML (pydantic-validated)
src/
  shared_kernel/          # pure transforms/adapters shared across slices
    Config_Adapter.py       # loads + validates config/*.yaml -> get_config()
    AudioIO_Adapter.py      # FLAC/WAV/OGG load (soundfile, path or bytes) + resample + manifest loader
    LogMel_Transform.py     # 80-bin log-mel frontend
    Tokenizer_Adapter.py    # SentencePiece BPE-500 wrapper
    BiasNorm.py, SwiGluFfn.py, RoPE_Transform.py   # SOTA blocks shared by encoder + LM
    Checkpoint_Adapter.py, MaskUtils.py, Logging_Adapter.py
  slices/
    BuildManifest/          # LibriSpeech split → manifest.jsonl; train BPE tokenizer
    ComputeCmvn/            # global mean/var over train → data/features/cmvn.pt
    ExtractFeatures/        # SpecAugment + speed-perturb dataset/collator/sampler
    TrainAcousticModel/     # Zipformer encoder + CTC head + attention decoder + Stage-A/B trainers
    TrainLanguageModel/     # STREAM-LM: causal GQA Transformer + corpus prep + LM trainer
    Decode/                 # streaming/offline two-pass CTC→attention rescore + LM fusion/rescore; StreamingSession (live)
    Evaluate/               # corpus WER/CER/RTF/latency + ablation table (jiwer) + dev α-tune
    Demo/                   # local FastAPI web UI: file upload + live-mic streaming transcription
scripts/verify_env.py     # asserts Blackwell + working torch
scripts/download_lm_text.py  # fetch the LibriSpeech-LM corpus for STREAM-LM
tests/                    # shape / round-trip / equivalence / count sanity tests
data/                     # LibriSpeech splits, manifests, tokenizer, cmvn, checkpoints, lm_data (gitignored)
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

Full retrain order after the value-residual change (Stage A → Stage B; STREAM-LM is unchanged, so
skip it unless you deleted its checkpoint):

```bash
# Stage A — Zipformer + CTC, now with value residual (~120k steps) → data/checkpoints/stage_a_last.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_a

# Stage B — hybrid CTC/attention (U2++ bidirectional decoder + joint loss + dynamic-chunk).
# Warm-starts the encoder + CTC head from stage_a_last.pt → data/checkpoints/stage_b_best.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_b

# Monitor either run (separate terminal): loss / lr / dev WER / dev blank_frac
.venv/bin/tensorboard --logdir runs/stage_b

# STREAM-LM (already trained, lm_best.pt kept — only rerun if you removed it). Download the
# LibriSpeech-LM corpus, pack it to uint16 bins (streamed to disk, bounded RAM), then train.
.venv/bin/python scripts/download_lm_text.py    # then gunzip; pack via PrepareLmData (see COMMANDS.md Step 6)
.venv/bin/python -m src.slices.TrainLanguageModel.train_lm
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
left-context trim, language-model weight) live in `config/decode.yaml`. Setting `lm_weight > 0`
turns on STREAM-LM shallow fusion in the first-pass beam plus n-best LM rescore in the second pass;
`lm_weight: 0.0` (the default) is byte-identical to the pre-LM decoder.

> The two-pass decoder is code-complete and streaming/offline equivalent, but the value-residual
> encoder + decoder need the Stage-A→Stage-B retrain above before `stage_b_best.pt` exists again.
> STREAM-LM (`lm_best.pt`, val ppl **16.4**) is unchanged and ready.

## Evaluation

The `Evaluate` slice reports corpus WER/CER plus mean RTF and first-partial latency across an
ablation of the two-pass decoder (`ctc_greedy → prefix_beam → attn_rescore → lm_rescore →
lm_fusion`, each × offline/streaming), writing `runs/eval/report.json`. Runs execute **two at a
time** on the GPU (offline + streaming of a stage in parallel, and the α-grid two-at-a-time under
`--tune`), so the ablation table and the sweep finish roughly twice as fast.

```bash
# Acoustic-only (LM off): the two-pass ablation on test-clean.
PYTHONPATH=. .venv/bin/python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl

# With the LM: sweep the fusion weight α on dev, freeze the best, then run the test table — in
# one command. Tuning on dev (never on test) keeps the headline number an honest held-out result.
PYTHONPATH=. .venv/bin/python -m src.slices.Evaluate.evaluate \
  data/manifests/test.jsonl --tune data/manifests/dev.jsonl
```

The LM only contributes at α (`lm_weight`) `> 0`; at α = 0 the `lm_*` stages equal `attn_rescore`
exactly and the run warns that they are inactive, so the report never silently misleads. The LM
prior is applied **exactly once** end to end: first-pass shallow fusion only *guides* the beam, and
the acoustic-only CTC score (not the fused score) is what the second pass rescores, so `lm_fusion`
no longer double-counts α. The second pass blends CTC against the attention score with
`rescore_ctc_weight`. Fresh `test-clean` numbers land after the retrain.

## Local demo (web UI)

Try the model by ear on your own machine — upload an audio file, or speak into the microphone and
watch partial transcripts stream in. The `Demo` slice is pure transport: it loads the model once and
drives the Decode slice (upload → full-context offline two-pass; live mic → streaming partials, then
a full-context final that replaces them on endpoint).

```bash
PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo   # then open http://127.0.0.1:8000
# --lm-weight 0.3 turns on LM shallow fusion; --checkpoint / --tokenizer / --host / --port also available
```

Binds `127.0.0.1` only (local, no auth); runs on GPU if available, else CPU. The browser captures at
16 kHz and sends raw PCM over a WebSocket for the live path — no FFmpeg or external assets needed.

## Tests

```bash
.venv/bin/python -m pytest -q          # expect 87 passed, 4 deselected (slow gates)
```

## Notes

- Audio decode uses `soundfile`, not `torchaudio.load`: torchaudio 2.11 removed its native
  decode/metadata backends and now routes through TorchCodec, which requires FFmpeg. torchaudio
  is kept only for pure-tensor ops (resample, mel spectrogram).
- Speed perturbation follows the Kaldi 3-way convention: a factor of 0.9 slows and lengthens
  the audio, while 1.1 speeds it up and shortens it.

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 PanzerPeter.
