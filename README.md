# STREAM

**S**peech **T**ranscription via **R**egularized **E**ncoder–**A**ttention **M**odeling.

A from-scratch, streaming-capable automatic speech recognition (ASR) system trained on the full
LibriSpeech **960 h** set (`train-clean-100 + train-clean-360 + train-other-500`). It targets a single
RTX 5070 (12 GB, Blackwell sm_120) and is built to stay hardware-efficient: state-of-the-art building
blocks reimplemented in pure PyTorch, with no k2/icefall dependency and no pretrained weights.

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

**Target from scratch:** offline WER ≈ 6–8 %, streaming ≈ 8–10 %, RTF < 0.3. This is not
Whisper-scale — the goal is the best result achievable from this data on this GPU.

## Status

| Plan | Scope | State |
|---|---|---|
| 1 | Data foundation: environment, manifests, tokenizer, features, dataloader | **Done** |
| 2 | Zipformer encoder + CTC head + Stage-A training | Code done; **retrain pending** — value residual added to the encoder |
| 3 | Causal encoder (Phase 0) + attention decoder + joint CTC/attn + Stage-B training (Phase 1) + streaming/offline decode (Phase 2) | Code done; **retrain pending** — value residual added to encoder + decoder |
| 4 | Neural LM (STREAM-LM) + two-pass LM fusion/rescore + Evaluate slice | **Done** — LM trained (val ppl **16.4**); needs LM retrain on the 960 h tokenizer |
| — | Local demo (web UI): upload a file or speak live | **Done** — `Demo` slice; verified end-to-end |

**Efficiency program (SP1–SP5).** A parallel track to reach 100 M-class WER from a ~45 M single-pass
streaming model, training *faster per run* on the one GPU:

| SP | Scope | State |
|---|---|---|
| 1 | 960 h data foundation: 5-split manifests, 960 h BPE retrain, fp16 log-mel mmap cache (~55 GB) | **Code-complete + reviewed** (branch `experimental`); user ran the build |
| 2 | Atomic SIGINT-safe **resumable** training harness (checkpoint + resume + `SignalGuard`) | **Code-complete + reviewed** |
| 3 | **Muon + AdamW + muP** optimizer stack (`config/optim.yaml`) — 2D hidden weights → Muon, rest → AdamW | **Code-complete + reviewed** |
| 4 | **BEST-RQ** self-supervised encoder pretrain → `bestrq_encoder.pt`, warm-starts Stage-A (`encoder_init`) | **Code-complete + reviewed** |
| 5 | Stateless RNN-T (transducer) head replacing the attention rescorer + InterCTC aux losses | Planned |

> **Retrain required.** The 960 h tokenizer retrain (SP1) plus the value-residual encoder/decoder
> change mean the Stage-A/B **and** LM checkpoints were removed and must be regenerated. The SP2
> checkpoint schema also stores an `"optimizers"` list, so any pre-SP2 `*.pt` fails to load. Prior WER
> figures (Stage-A dev 0.111, Stage-B dev 0.0999, test-clean 9.1 %/12.0 %) predate these changes and
> are superseded; new numbers land after the retrain below.

## Layout

The repository follows a vertical-slice layout adapted from `VSA.md` for a training pipeline.
Slices communicate only through artifact files (manifests, tokenizer, checkpoints) and typed
dataclass DTOs, never by importing each other's internals.

```
config/                   # audio/augment/model/training/decode/lm/eval/optim/pretrain tunables as YAML (pydantic-validated)
src/
  shared_kernel/          # pure transforms/adapters shared across slices
    Config_Adapter.py       # loads + validates config/*.yaml -> get_config()
    AudioIO_Adapter.py      # FLAC/WAV/OGG load (soundfile, path or bytes) + resample + manifest loader
    LogMel_Transform.py     # 80-bin log-mel frontend
    Tokenizer_Adapter.py    # SentencePiece BPE-500 wrapper
    BiasNorm.py, SwiGluFfn.py, RoPE_Transform.py   # SOTA blocks shared by encoder + LM
    Checkpoint_Adapter.py   # atomic stateful save/load + resume_if_available (SP2)
    SignalGuard.py          # cooperative SIGINT/SIGTERM stop for training loops (SP2)
    Muon_Optimizer.py, Optimizer_Adapter.py, mup.py   # Muon+AdamW+muP optimizer stack (SP3)
    RandomProjectionQuantizer.py   # frozen BEST-RQ target quantizer (SP4)
    MaskUtils.py, Logging_Adapter.py
  slices/
    BuildManifest/          # LibriSpeech split → manifest.jsonl; train BPE tokenizer
    ComputeCmvn/            # global mean/var over train → data/features/cmvn.pt
    ExtractFeatures/        # fp16 log-mel mmap cache + dataset/collator/sampler + GPU SpecAugment (SP1)
    PretrainEncoder/        # BEST-RQ self-supervised encoder pretrain → data/checkpoints/bestrq_encoder.pt (SP4)
    TrainAcousticModel/     # Zipformer encoder + CTC head + attention decoder + Stage-A/B trainers
    TrainLanguageModel/     # STREAM-LM: causal GQA Transformer + corpus prep + LM trainer
    Decode/                 # streaming/offline two-pass CTC→attention rescore + LM fusion/rescore; StreamingSession (live)
    Evaluate/               # corpus WER/CER/RTF/latency + ablation table (jiwer) + dev α-tune
    Demo/                   # local FastAPI web UI: file upload + live-mic streaming transcription
scripts/verify_env.py     # asserts Blackwell + working torch
scripts/build_manifests.py, scripts/train_tokenizer.py, scripts/precompute_features.py  # 960h build (SP1)
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

The 960 h build is four ordered scripts (see [COMMANDS.md](COMMANDS.md) Step 2 for details). The fp16
log-mel cache is the big one-time cost (~55 GB) and lets the training epoch loop run GPU-bound.

```bash
PYTHONPATH=. .venv/bin/python scripts/build_manifests.py      # 5-split manifests (train 281,241 utts)
PYTHONPATH=. .venv/bin/python scripts/train_tokenizer.py      # retrain BPE-500 on 960h (invalidates old LM)
PYTHONPATH=. .venv/bin/python scripts/compute_cmvn.py         # global CMVN over a 15% sample → data/features/cmvn.pt
PYTHONPATH=. .venv/bin/python scripts/precompute_features.py  # fp16 log-mel mmap cache → data/features/mel/
```

## Training

Full retrain order (Stage A → Stage B → STREAM-LM — all three checkpoints must be regenerated on the
960 h tokenizer):

```bash
# (Optional, SP4) BEST-RQ self-supervised encoder pretrain → data/checkpoints/bestrq_encoder.pt
.venv/bin/python -m src.slices.PretrainEncoder.pretrain_bestrq

# Stage A — Zipformer + CTC, with value residual (~120k steps) → data/checkpoints/stage_a_last.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_a
# …or warm-start the encoder from the BEST-RQ pretrain above (encoder_init on the command):
.venv/bin/python -c "from src.slices.TrainAcousticModel.StageATrainer_Handler import run_stage_a; from src.slices.TrainAcousticModel.StageATrainer_Command import StageATrainCommand; import dataclasses as d; run_stage_a(d.replace(StageATrainCommand(), encoder_init='data/checkpoints/bestrq_encoder.pt'))"

# Stage B — hybrid CTC/attention (U2++ bidirectional decoder + joint loss + dynamic-chunk).
# Warm-starts the encoder + CTC head from stage_a_last.pt → data/checkpoints/stage_b_best.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_stage_b

# Monitor any run (separate terminal): loss / lr / dev WER / dev blank_frac
.venv/bin/tensorboard --logdir runs/stage_b

# STREAM-LM — must retrain (the 960h tokenizer invalidated the old lm_best.pt). Download the
# LibriSpeech-LM corpus, pack it to uint16 bins (streamed to disk, bounded RAM), then train.
.venv/bin/python scripts/download_lm_text.py    # then gunzip; pack via PrepareLmData (see COMMANDS.md Step 6)
.venv/bin/python -m src.slices.TrainLanguageModel.train_lm
```

**Resumable & SIGINT-safe (SP2).** Every trainer (pretrain, Stage-A, Stage-B) atomically checkpoints
`*_last.pt` (model + all optimizers + RNG + step) and **auto-resumes** from it — just re-launch the
same command after an interrupt or crash. Ctrl-C is caught cooperatively (finishes the step,
checkpoints, exits clean). Force a fresh run with `resume=False` on the command.

All tunables are read from `config/*.yaml` via `get_config()` — edit the YAML, no code change
required. The optimizer stack (Muon+AdamW+muP) lives in `config/optim.yaml`, BEST-RQ pretrain in
`config/pretrain.yaml`.

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
> STREAM-LM must also be retrained on the 960 h tokenizer (the old `lm_best.pt` is now incompatible).

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
.venv/bin/python -m pytest -q          # expect 129 passed, 4 deselected (slow gates)
```

## Notes

- Audio decode uses `soundfile`, not `torchaudio.load`: torchaudio 2.11 removed its native
  decode/metadata backends and now routes through TorchCodec, which requires FFmpeg. torchaudio
  is kept only for pure-tensor ops (resample, mel spectrogram).
- Features are precomputed once into an fp16 log-mel mmap cache (SP1); SpecAugment is a GPU batch op
  (`SpecAugmentBatch.py`, built in SP1 but not yet wired into the trainers — that's an SP5 task).
  Speed perturbation was dropped (its coprime-resample cost starved the GPU).

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 PanzerPeter.
