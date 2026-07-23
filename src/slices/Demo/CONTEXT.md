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

Every transcript leaving the server passes through `TranscriptFormat.format_transcript`: the
tokenizer is trained on LibriSpeech's upper-case unpunctuated text, so a raw decode reads
`MISTER QUILTER IS THE APOSTLE`. The pass lower-cases, restores the leading capital and the pronoun
"I", and stops there — proper nouns and sentence boundaries are unrecoverable from unpunctuated
output, and guessing them would misreport what the model said. It is **display-only**: the Decode
slice still emits corpus-cased text, so Evaluate's WER stays comparable.

Live partials come from the causal streaming encoder + greedy RNN-T decoding (`StreamingSession`),
so they appear mid-utterance; on endpoint the partials are replaced by the authoritative
full-context beam-search result. The browser captures at a 16 kHz `AudioContext` (no server-side
resampling on the live path).

## Artifacts

- Consumes: `data/checkpoints/transducer_best.pt`, `data/tokenizer/bpe500.model` (and
  `data/checkpoints/lm_best.pt` when `--lm-weight > 0` turns on n-best rescoring).
- Produces: nothing on disk — an interactive service.

## Entry point

`PYTHONPATH=. .venv/bin/python -m src.slices.Demo.serve_demo --lm-weight 0.6 --ilm-weight 0.2`
→ open `http://127.0.0.1:8000`. Flags: `--checkpoint`, `--tokenizer`, `--host`, `--port`,
`--lm-weight` (alpha), `--ilm-weight` (beta, the ILME subtraction), `--beam-size`. The two weights
default to `config/decode.yaml`, whose committed values are the alpha=0 regression lock — pass the
pair tuned by `evaluate.py --tune` (0.6 / 0.2) to hear the configuration the reported WER was
measured at. Startup prints the resolved beam/LM settings so a silent fallback to acoustic-only is
visible. Binds `127.0.0.1` only (no auth). Runs on GPU if available, else CPU; the model is held
resident for the process lifetime.
