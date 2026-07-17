# Demo Slice

A local, single-user web demo for trying the trained model by ear: upload an audio file, or speak
into the microphone and watch partial transcripts stream in. It is **pure transport + composition**
— it owns no ASR logic. It loads the model once and drives the Decode slice's public entry points,
exactly as the Evaluate slice does (the model definitions + checkpoints are the artifact contract;
no trainer or decode internals are imported).

## Endpoints

| Route | Method | Flow |
|---|---|---|
| `/` | GET | serves `static/index.html` (self-contained UI, inline CSS/JS) |
| `/transcribe` | POST (multipart) | uploaded WAV/FLAC/OGG → `load_audio_bytes` → `StreamingDecoder_Handler.decode_waveform(streaming=False)` (full-context single-pass RNN-T beam search, best WER) → `{text, rtf, seconds}` |
| `/stream` | WebSocket | binary 16 kHz mono float32 PCM frames → `StreamingSession.accept_audio` → `{partial}`; a text `__eof__` frame → `StreamingSession.finalize()` (offline re-decode) → `{final, rtf}` |

Live partials come from the causal streaming encoder + greedy RNN-T decoding (`StreamingSession`),
so they appear mid-utterance; on endpoint the partials are replaced by the authoritative
full-context beam-search result. The browser captures at a 16 kHz `AudioContext` (no server-side
resampling on the live path).

## Artifacts

- Consumes: `data/checkpoints/transducer_best.pt`, `data/tokenizer/bpe500.model` (and
  `data/checkpoints/lm_best.pt` when `--lm-weight > 0` enables shallow fusion).
- Produces: nothing on disk — an interactive service.

## Entry point

`PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo` → open `http://127.0.0.1:8000`.
Flags: `--checkpoint`, `--tokenizer`, `--host`, `--port`, `--lm-weight`. Binds `127.0.0.1` only (no
auth). Runs on GPU if available, else CPU; the model is held resident for the process lifetime.
