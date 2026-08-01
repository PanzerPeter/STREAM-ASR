# Demo

## Purpose
Serve a local web UI for trying the trained model by ear: upload an audio file, or speak into the
microphone and watch partial transcripts stream in.

## Entry Point
- Type: CLI (`serve_demo.py`) → `DemoServer_Handler.build_app` (FastAPI/uvicorn)
- Input: CLI flags `--checkpoint`, `--tokenizer`, `--host`, `--port`, `--lm-weight` (alpha),
  `--ilm-weight` (beta, the ILME subtraction), `--beam-size`
- Output: an HTTP/WebSocket service on `http://127.0.0.1:8000`

| Route | Method | Flow |
|---|---|---|
| `/` | GET | serves `static/index.html` (self-contained UI, inline CSS/JS) |
| `/config` | GET | the resolved decode settings → `{version, device, checkpoint, beam_size, lm, chunk_ms}` (`lm` is `null` when no LM is attached); read-only |
| `/transcribe` | POST (multipart) | uploaded WAV/FLAC/OGG → `load_audio_bytes` → `StreamingDecoder_Handler.decode_waveform(streaming=False)` (full-context single-pass RNN-T beam search, best WER) → `{text, rtf, seconds, decode_s}`; uploads over 64 MiB are rejected 413 on the read, since a file is held whole in memory before it can be decoded |
| `/stream` | WebSocket | binary 16 kHz mono float32 PCM frames → `StreamingSession.accept_audio` → `{partial}`; a text `__eof__` frame → `StreamingSession.finalize()` (offline re-decode) → `{final, rtf}` |

## Data Ownership
- Consumes artifacts: `data/checkpoints/transducer_avg.pt`, `data/tokenizer/bpe500.model` (and
  `data/checkpoints/lm_best.pt` when `--lm-weight > 0` turns on n-best rescoring).
- Produces: nothing on disk, since it is an interactive service.

## Shared Kernel
- `Config_Adapter.get_config().decode`: defaults for the beam and rescoring weights.
- `AudioIO_Adapter.load_audio_bytes`: decoding an uploaded file held in memory.

## Notes
The slice is pure transport and composition, and owns no ASR logic. It loads the model once and
drives the Decode slice's public entry points, exactly as Evaluate does (the model definitions and
checkpoints are the artifact contract; no trainer or decode internals are imported).

Every transcript leaving the server passes through `TranscriptFormat.format_transcript`: the
tokenizer is trained on LibriSpeech's upper-case unpunctuated text, so a raw decode reads
`MISTER QUILTER IS THE APOSTLE`. The pass lower-cases, restores the leading capital and the pronoun
"I", and stops there, because proper nouns and sentence boundaries are unrecoverable from
unpunctuated output and guessing them would misreport what the model said. It is display-only: the
Decode slice still emits corpus-cased text, so Evaluate's WER stays comparable.

Live partials come from the causal streaming encoder plus greedy RNN-T decoding
(`StreamingSession`), so they appear mid-utterance. On endpoint the partials are replaced by the
authoritative full-context beam-search result. The browser captures at a 16 kHz `AudioContext`,
which keeps resampling off the live server path. The capture graph ends in a **muted** gain node:
a `ScriptProcessor` only fires while something downstream pulls it, but pulling through
`ctx.destination` plays the microphone back through the speakers, which the model then transcribes
alongside the speaker (and howls without headphones). A level meter reads the same stream through an
`AnalyserNode`, because otherwise an empty transcript cannot be told apart from a dead microphone.

The page opens with `/config` and renders it as a strip of chips above the cards. `serve_demo`
prints the same line at startup, which is invisible to whoever is looking at the browser: a run that
fell back to the `alpha = 0` regression lock, or to `transducer_best.pt` instead of the averaged
checkpoint, only sounds slightly worse. The strip is display-only, and the weights stay CLI flags,
so what the page reports is always what the process was started with.

The weight flags default to `config/decode.yaml`, whose committed values are the `alpha = 0`
regression lock. Pass the dev-tuned pair (`--lm-weight 0.6 --ilm-weight 0.2`, from `evaluate.py`'s
automatic sweep) to hear the configuration the reported WER was measured at. Startup prints the
resolved beam/LM settings, which makes a silent fallback to acoustic-only visible. Binds
`127.0.0.1` only, with no auth. Runs on GPU if available, else CPU, and the model is held resident
for the process lifetime.
