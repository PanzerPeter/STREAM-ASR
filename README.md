# STREAM

**S**peech **T**ranscription via **R**egularized **E**ncoder–**A**coustic **M**odeling.

A from-scratch, streaming-capable automatic speech recognition (ASR) system trained on the full
LibriSpeech **960 h** set (`train-clean-100 + train-clean-360 + train-other-500`). It targets a single
RTX 5070 (12 GB, Blackwell sm_120) and is built to stay hardware-efficient: state-of-the-art building
blocks reimplemented in pure PyTorch, with no k2/icefall dependency and no pretrained weights.

**Result (test-clean, 2,620 utts):** offline **4.30 % WER** (CER 1.53 %) / streaming **5.73 % WER**
(CER 2.13 %) with the RNN-T beam + LM rescoring — real-time factor ≪ 1 on the one GPU. Both land
well inside the design targets. See [Evaluation](#evaluation).

## Approach

A single-pass streaming RNN-T (transducer) system built around a Zipformer acoustic encoder:

| Module | Component | Role |
|---|---|---|
| Acoustic encoder | Zipformer (~53.8 M params) | log-mel → conv subsample → multi-rate Zipformer stacks with dynamic-chunk masking (~25 Hz output) |
| Prediction network | `StatelessPredictor` | icefall-style: embeds the previous non-blank token + a small causal depthwise-conv context — no recurrence, so streaming state is just the last `context-1` token ids |
| Joint network | `TransducerJoiner` | additive joiner (project encoder + predictor, sum, tanh, readout to vocab+blank); trains against the full `[B,T,U+1,V]` lattice, decodes one `(t,u)` cell at a time |
| Auxiliary heads | CTC head + InterCTC taps | `ctc_aux_weight * CTC` on the final encoder output plus weighted CTC losses tapped after two intermediate stacks — regularizers and a cheap greedy-WER health probe, not a separate decoding pass |
| Language model | Causal Transformer LM (STREAM-LM) | optional **n-best rescoring** of the acoustic beam: re-rank each hypothesis by `acoustic + α·lm_seq`, one full-sequence LM forward per hypothesis (`decode.lm_weight`, α) |

The model is ~55.3 M params total. A single set of encoder weights runs either offline (full
context, best WER) or streaming (small chunk, low latency) through dynamic-chunk masking, and one
joint training stage (warm-started from `bestrq_encoder.pt`) produces the checkpoint both modes
decode from. The acoustic decode is a single RNN-T beam pass; the LM (when attached) only re-ranks
that beam's n-best — there is no second acoustic model.

**Value residual (ResFormer/SVFormer).** The Zipformer encoder stacks and STREAM-LM use
value-residual attention: block/layer 0 of each stack injects its attention *values* into the
deeper blocks, adding a gradient shortcut through attention that trains stably at depth so the same
accuracy is reachable at a narrower width. The mix is a **learnable per-block gate initialised to
0** (`encoder_value_residual_lambda` in `config/model.yaml`, `value_residual_lambda` in
`config/lm.yaml`): a fresh model therefore starts *identical* to a no-value-residual baseline and
the residual grows only as far as training wants it. This matters because the CTC branch (main +
InterCTC) sits on a blank-collapse knife-edge — a fixed non-zero gate destabilises the escape,
whereas the zero-init gate does not. The encoder's streaming path caches post-residual values, so
`streaming_forward` stays exactly equal to the chunked `forward`
(`test_streaming_forward_equivalence`). The transducer's own `StatelessPredictor`/`TransducerJoiner`
are conv/linear-based (no attention), so value residual does not apply to them.

**Not Whisper-scale by design** — the goal is the best result achievable from this data on this GPU,
and the from-scratch target (offline 6–8 % / streaming 8–10 % WER at RTF < 0.3) is beaten on both
paths.

## Status

Complete and GPU-validated end-to-end. The pipeline — 960 h data build → BEST-RQ encoder pretrain →
single joint transducer stage (120 k steps, best dev transducer-WER **0.0637**) → STREAM-LM
(40 k steps, val ppl **16.19**) → `test-clean` eval — runs on one RTX 5070 and produces the
[Evaluation](#evaluation) numbers. Training uses a resumable, SIGINT-safe harness and a
Muon+AdamW+muP optimizer stack; all tunables live in `config/*.yaml`. The local
[demo](#local-demo-web-ui) serves the model over a browser UI.

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
    Checkpoint_Adapter.py   # atomic stateful save/load + resume_if_available
    SignalGuard.py          # cooperative SIGINT/SIGTERM stop for training loops
    Muon_Optimizer.py, Optimizer_Adapter.py, mup.py   # Muon+AdamW+muP optimizer stack
    RandomProjectionQuantizer.py   # frozen BEST-RQ target quantizer
    MaskUtils.py, Logging_Adapter.py
  slices/
    BuildManifest/          # LibriSpeech split → manifest.jsonl; train BPE tokenizer
    ComputeCmvn/            # global mean/var over train → data/features/cmvn.pt
    ExtractFeatures/        # fp16 log-mel mmap cache + dataset/collator/sampler + GPU SpecAugment
    PretrainEncoder/        # BEST-RQ self-supervised encoder pretrain → data/checkpoints/bestrq_encoder.pt
    TrainAcousticModel/     # Zipformer encoder + CTC/InterCTC heads + StatelessPredictor + TransducerJoiner + transducer trainer
    TrainLanguageModel/     # STREAM-LM: causal GQA Transformer + corpus prep + LM trainer
    Decode/                 # streaming/offline single-pass RNN-T beam search + LM n-best rescoring; StreamingSession (live)
    Evaluate/               # corpus WER/CER/RTF/latency + ablation table (jiwer) + dev α-tune
    Demo/                   # local FastAPI web UI: file upload + live-mic streaming transcription
scripts/verify_env.py     # asserts Blackwell + working torch
scripts/build_manifests.py, scripts/train_tokenizer.py, scripts/precompute_features.py  # 960h build
scripts/download_lm_text.py  # fetch the LibriSpeech-LM corpus for STREAM-LM
scripts/average_checkpoints.py  # mean the tail of transducer_step*.pt snapshots into one decode checkpoint
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
PYTHONPATH=. .venv/bin/python scripts/train_tokenizer.py      # train BPE-500 on 960h transcripts
PYTHONPATH=. .venv/bin/python scripts/compute_cmvn.py         # global CMVN over a 15% sample → data/features/cmvn.pt
PYTHONPATH=. .venv/bin/python scripts/precompute_features.py  # fp16 log-mel mmap cache → data/features/mel/
```

## Training

Order: BEST-RQ pretrain → single joint transducer stage → STREAM-LM (both checkpoints train against
the 960 h tokenizer):

```bash
# (Optional) BEST-RQ self-supervised encoder pretrain → data/checkpoints/bestrq_encoder.pt
.venv/bin/python -m src.slices.PretrainEncoder.pretrain_bestrq

# Transducer — single joint training stage: Zipformer encoder + StatelessPredictor + TransducerJoiner,
# trained with rnnt + ctc_aux_weight*ctc + interctc losses (~120k steps). Warm-starts the encoder from
# data/checkpoints/bestrq_encoder.pt by default (training.transducer.warm_start)
# → data/checkpoints/transducer_last.pt / transducer_best.pt
.venv/bin/python -m src.slices.TrainAcousticModel.train_transducer

# Monitor any run (separate terminal): loss / lr / dev transducer-WER / dev blank_frac
.venv/bin/tensorboard --logdir runs/transducer

# STREAM-LM — trained on the 960h tokenizer (val ppl 17.2). Download the LibriSpeech-LM corpus,
# pack it to uint16 bins (streamed to disk, bounded RAM), then train.
.venv/bin/python scripts/download_lm_text.py    # then gunzip; pack via PrepareLmData (see COMMANDS.md Step 6)
.venv/bin/python -m src.slices.TrainLanguageModel.train_lm
```

**Resumable & SIGINT-safe.** Every trainer (BEST-RQ pretrain, transducer) atomically
checkpoints `*_last.pt` (model + all optimizers + RNG + step) and **auto-resumes** from it — just
re-launch the same command after an interrupt or crash. Ctrl-C is caught cooperatively (finishes
the step, checkpoints, exits clean). Force a fresh run with `resume=False` on the command.

**Checkpoint averaging.** The transducer trainer keeps a rolling window of
`transducer_step{N}.pt` snapshots (`training.transducer.keep_last_n`, default 5). Mean the tail into
one decode checkpoint and point `config/decode.yaml` / `config/eval.yaml` at it:

```bash
PYTHONPATH=. .venv/bin/python scripts/average_checkpoints.py --last-n 5   # → data/checkpoints/transducer_avg.pt
```

All tunables are read from `config/*.yaml` via `get_config()` — edit the YAML, no code change
required. The optimizer stack (Muon+AdamW+muP) lives in `config/optim.yaml`, BEST-RQ pretrain in
`config/pretrain.yaml` (`mask_prob: 0.05` × `mask_span: 10` ≈ 39 % of frames masked — the SSL task
must be hard enough to force useful features).

## Streaming / Offline Decode

The encoder is causal-in-time: causal `Conv2dSubsampling` frontend, per-frame `ConvModule` with
`BiasNorm`, and RoPE `pos_offset`. This enables streaming inference through
`ZipformerEncoder.streaming_forward()` with exact equivalence to full-context batched `forward()`.

```bash
# Offline (single-pass RNN-T beam over full-context encoder memory): test-clean 4.3–5.1 % WER
PYTHONPATH=. .venv/bin/python -m src.slices.Decode.streaming_decode data/Val/dev-clean-0001.flac --offline

# Streaming (single-pass RNN-T beam over chunked, cached encoder memory): test-clean 5.7–7.0 %, low latency
PYTHONPATH=. .venv/bin/python -m src.slices.Decode.streaming_decode data/Val/dev-clean-0001.flac
```

Both modes use `data/checkpoints/transducer_best.pt` as the default checkpoint and
`data/tokenizer/bpe500.model` for tokenization. Decoding parameters (chunk size, beam width,
left-context trim, language-model weight) live in `config/decode.yaml`. Setting `lm_weight > 0`
turns on STREAM-LM **n-best rescoring** of the acoustic beam (the tuned optimum is **α = 0.4**);
`lm_weight: 0.0` (the shipped default) is byte-identical to the pre-LM decoder. `length_bonus`
adds a per-token re-ranking bonus to counter RNN-T's un-normalised deletion bias (default `0.0`,
swept alongside α by `evaluate.py --tune`).

Hypotheses that share a token prefix are **recombined** (log-sum-exp merged) inside the beam, so the
beam width buys distinct transcripts instead of duplicate alignments of the same one.

## Evaluation

The `Evaluate` slice reports corpus WER/CER plus mean RTF and first-partial latency across an
ablation of the single-pass transducer decoder (`greedy_transducer → beam → beam_lm`, each ×
offline/streaming), writing `runs/eval/report.json`. Runs execute **two at a time** on the GPU
(offline + streaming of a stage in parallel, and the α-grid two-at-a-time under `--tune`), so the
ablation table and the sweep finish roughly twice as fast.

**test-clean results (n = 2,620, tuned α = 0.4).** Search and the LM both help monotonically:

| Stage | Offline WER / CER | Streaming WER / CER | RTF (off / stream) |
|---|---|---|---|
| `greedy_transducer` | 5.41 % / 1.90 % | 7.34 % / 2.67 % | 0.015 / 0.051 |
| `beam` | 5.12 % / 1.76 % | 6.97 % / 2.48 % | 0.060 / 0.099 |
| **`beam_lm`** | **4.30 % / 1.53 %** | **5.73 % / 2.13 %** | 0.115 / 0.160 |

Streaming latency ≈ 27 ms (first partial). The streaming↔offline gap is **1.4 abs pts** — the cost
of the chunked causal encoder. The LM is worth −0.82 abs offline / −1.24 abs streaming on top of the
beam; beam over greedy is worth another −0.29 / −0.37. RTFs are measured with the offline and
streaming passes sharing the GPU, so each is a pessimistic bound on a dedicated run.

```bash
# Acoustic-only (LM off): the single-pass transducer ablation on test-clean.
PYTHONPATH=. .venv/bin/python -m src.slices.Evaluate.evaluate data/manifests/test.jsonl

# With the LM: sweep the fusion weight α on dev, freeze the best, then run the test table — in
# one command. Tuning on dev (never on test) keeps the headline number an honest held-out result.
PYTHONPATH=. .venv/bin/python -m src.slices.Evaluate.evaluate \
  data/manifests/test.jsonl --tune data/manifests/dev.jsonl
```

The LM only contributes at α (`lm_weight`) `> 0`; at α = 0 the `beam_lm` stage equals `beam`
exactly and the run warns that it is inactive, so the report never silently misleads. The LM prior
is applied by **n-best rescoring** — the acoustic beam runs once, then each hypothesis is re-ranked
by `acoustic + α·lm_seq` (one full-sequence LM forward per hypothesis). Tuning decodes dev once
acoustic-only and sweeps α over the cached scores for free, which is why the whole eval finishes in
a few GPU-hours rather than a day. The 2026-07-20 run selected **α = 0.4** (dev WER 0.0469, against
0.0534 acoustic-only).

## Local demo (web UI)

Try the model by ear on your own machine — upload an audio file, or speak into the microphone and
watch partial transcripts stream in. The `Demo` slice is pure transport: it loads the model once and
drives the Decode slice (upload → full-context offline single-pass RNN-T beam; live mic → streaming
partials, then a full-context final that replaces them on endpoint).

```bash
PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo   # then open http://127.0.0.1:8000
# --lm-weight 0.4 turns on LM n-best rescoring (the tuned optimum); --checkpoint / --tokenizer /
# --host / --port also available. Omit --lm-weight for the faster acoustic-only decoder.
```

Binds `127.0.0.1` only (local, no auth); runs on GPU if available, else CPU. The browser captures at
16 kHz and sends raw PCM over a WebSocket for the live path — no FFmpeg or external assets needed.

## Tests

```bash
.venv/bin/python -m pytest -q          # expect 144 passed, 2 deselected (slow gates)
```

## Notes

- Audio decode uses `soundfile`, not `torchaudio.load`: torchaudio 2.11 removed its native
  decode/metadata backends and now routes through TorchCodec, which requires FFmpeg. torchaudio
  is kept only for pure-tensor ops (resample, mel spectrogram).
- Features are precomputed once into an fp16 log-mel mmap cache; SpecAugment is a GPU batch op
  (`SpecAugmentBatch.py`) wired into `TransducerModel.joint_loss` on the train path only, gated by
  `training.spec_augment` (default `true`).

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 PanzerPeter.
