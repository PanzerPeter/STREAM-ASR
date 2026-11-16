# STREAM ASR

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20%2B%20CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)

**S**peech **T**ranscription via **R**egularized **E**ncoder-**A**coustic **M**odeling: a
from-scratch streaming speech recognizer for English, trained end to end on LibriSpeech 960 h on a
single 12 GB consumer GPU.

* **One model, two modes.** The same weights decode offline and streaming (320 ms chunks).
* **Small and fast.** 36.7 M parameters; streaming real-time factor 0.05 with beam search and LM
  rescoring, 80 ms median emission latency, and a constant 5.3 MB streaming state for any stream
  length.
* **Cheap to train.** 15.1 h from scratch on one RTX 5070, with no pretrained weights and no
  external data in the acoustic model.
* **Pure PyTorch.** No k2, icefall or external decoder: the banded RNN-T loss, Muon optimizer,
  label-looping decoder and beam search are all in this repository.

## Results

LibriSpeech test WER, v2.0 release (`transducer_avg.pt` + STREAM-LM). Rescoring weights are tuned on
the matching dev set, never on test.

| Decoding | test-clean offline | test-clean streaming | test-other offline | test-other streaming |
|---|---|---|---|---|
| greedy | 3.17 % | 4.24 % | 7.91 % | 10.92 % |
| beam 10 | 3.15 % | 4.24 % | 7.76 % | 10.48 % |
| beam 10 + LM + ILME | **2.64 %** | **3.18 %** | **6.51 %** | **8.87 %** |

Streaming is 320 ms chunks with 5.12 s of left context. On one RTX 5070: RTF 0.019 offline and 0.049
streaming with the LM, 13 ms to first partial, 0.12 s to finalize. Full tables, the budget curve and
the tuned weights are in [MODEL_ARCHITECTURE.md §12](MODEL_ARCHITECTURE.md#12-results).

## Quickstart: transcribe with the released weights

```bash
git clone --branch v2.0 https://github.com/PanzerPeter/STREAM-ASR.git && cd STREAM-ASR
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

V=release/stream-asr-v2.0
gh release download v2.0 -D $V -p '*.safetensors'
(cd $V && sha256sum -c SHA256SUMS)

.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model             # streaming
.venv/bin/python -m src.slices.Decode.streaming_decode AUDIO.flac --offline \
    --checkpoint $V/model.safetensors --tokenizer $V/bpe500.model             # offline
```

Input is 16 kHz mono speech. [release/stream-asr-v2.0/README.md](release/stream-asr-v2.0/README.md)
is the model card: files, LM rescoring, the local web demo, and limitations.

## Train from scratch

Needs the LibriSpeech corpus under `data/{Train,Val,Test}`, about 250 GB of disk (the corpus plus a
157 GiB feature cache) and a CUDA GPU. The recipe was built on an RTX 5070 (12 GB, Blackwell
sm_120, CUDA 12.8 wheels); `verify_env.py` checks for exactly that card, so edit its expected
capability on other hardware and re-run `scripts/profile_transducer_step.py` before trusting the
batch budgets.

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/verify_env.py

# data: manifests, BPE-500 tokenizer, CMVN, 2x speed-perturbed manifest, fp16 mel cache
PYTHONPATH=. python scripts/build_manifests.py
PYTHONPATH=. python scripts/train_tokenizer.py
PYTHONPATH=. python scripts/compute_cmvn.py
PYTHONPATH=. python scripts/build_speed_perturb_manifest.py
PYTHONPATH=. python scripts/precompute_features.py

# transducer: 600k batches, ~15 h on an RTX 5070, then average the last five snapshots
python -m src.slices.TrainAcousticModel.train_transducer
PYTHONPATH=. python scripts/average_checkpoints.py --last-n 5

# STREAM-LM for rescoring, then evaluate (tunes the LM weights on dev, reports on test)
python -m src.slices.TrainLanguageModel.train_lm
python -m src.slices.Evaluate.evaluate --clean
python -m src.slices.Evaluate.evaluate --other
```

[COMMANDS.md](COMMANDS.md) has every command and flag, the LM data preparation, and the failure
modes worth knowing first.

## How it works

```
raw log-mel [B, T, 80] ──► CMVN, mean fill past each length, pad to 16 frames
  └► DwSepSubsampling (x4, stride-aligned causal pad) ──► rms_norm            25 Hz, d 256
       └► S0: 4 blocks                                                        25 Hz
            └► pool ► S1: 8 blocks ─────────────────────── skip ──┐          12.5 Hz, d 384
                 └► pool ► S2: linear path ► upsample ► (+) ◄─────┘          6.25 Hz, d 512
                                              └► S3: 8 blocks ► rms_norm      12.5 Hz memory
memory ─► CTC head (sets the band, auxiliary CTC loss)
       └► additive joiner on the band ◄── stateless predictor (BPE-500 + blank)
```

* **Encoder.** A three-rate U-Net with 20 blocks. Each block is gated QK-norm RoPE attention, then a
  conv-gated MLP on its result, each a residual step `x + β ⊙ rms_norm(y)` with a per-channel gate
  `β = 0.05 + 0.95 σ(b)`.
* **Amplitude bounded by construction.** Every stack entry and the output are weightless RMSNorms and
  the floored gates are the only learnable residual-path scale, so no activation can grow without
  bound and no parameter is ever projected or clamped.
* **Banded RNN-T.** After a 10k-batch warmup the joiner and loss run on 17 lattice cells per frame
  around the CTC head's cumulative token count, repaired so both lattice terminals stay inside.
  The band's loss gap to the full lattice measured 0.0 throughout training.
* **Dynamic-chunk training.** Each batch samples a chunk size (full, 320 ms, 640 ms) and a left
  context (2.56 s, 5.12 s, unlimited), so one model serves both modes. Streaming decode is equal to
  the batched forward to fp32 round-off.
* **Decoding.** Exact label-looping greedy, a time-synchronous beam with recombination, and n-best
  rescoring with STREAM-LM minus an internal-LM estimate.

[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) covers every operator, the training recipe, the
invariants and how the recipe was chosen.

## Repository layout

```
config/                 pydantic-validated YAML, one file per section (authoritative values)
src/shared_kernel/      pure transforms + infra adapters (banded RNN-T loss, Muon, RoPE, IO, ...)
src/slices/
  BuildManifest/        manifests, 2x speed-perturbed manifest, BPE-500 tokenizer
  ComputeCmvn/          global mean/std over a train sample
  ExtractFeatures/      fp16 mel cache, dataset, frame-bucket sampler, SpecAugment (GPU)
  TrainAcousticModel/   S2 encoder, predictor, joiner, banded RNN-T + CTC trainer
  TrainLanguageModel/   STREAM-LM, the n-best rescorer
  Decode/               streaming / offline search + LM rescoring
  Evaluate/             WER/CER/RTF, edge errors, emission latency, dev-tuned weights
  Demo/                 local FastAPI upload + live-microphone UI
scripts/                data prep, checkpoint averaging, safetensors export, step profiler
tests/                  CPU suite (pytest), GPU gates under -m slow
release/                model cards, tokenizer and checksums per released version
```

The code follows Vertical Slice Architecture ([VSA.md](VSA.md)): slices talk only through artifact
files and frozen command DTOs, and each slice's `CONTEXT.md` is its contract.

## License and attribution

STREAM ASR is open source under the [Apache License 2.0](LICENSE). Use it, modify it and ship it,
personally, academically or commercially, in open- or closed-source products, with no permission
needed and no royalty owed.

One thing is asked in return, and the license makes it binding (§4(d)): carry the [NOTICE](NOTICE)
file into anything you distribute that is built on this work, and credit it where you list your
other open-source components:

> Based on STREAM ASR (https://github.com/PanzerPeter/STREAM-ASR), Copyright 2026 PanzerPeter,
> licensed under Apache-2.0.

That covers the code, the trained checkpoints and the derived data artifacts alike. LibriSpeech is
distributed by its own authors under CC BY 4.0 and is not part of this repository.
