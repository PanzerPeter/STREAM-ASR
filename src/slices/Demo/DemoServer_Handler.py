# src/slices/Demo/DemoServer_Handler.py — FastAPI app for the local demo.
# Pure transport + composition: it owns no ASR logic, only wiring HTTP/WebSocket I/O to the Decode
# slice's public handler (file upload -> full-context single-pass RNN-T beam search) and
# StreamingSession (live mic -> partials then an offline final). Consistent with how
# src/slices/Evaluate composes the handler. Decoded text is sentence-cased on the way out
# (TranscriptFormat) -- a display concern, so it stays out of the scored Decode path.
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from src.shared_kernel.AudioIO_Adapter import load_audio_bytes
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingSession import StreamingSession
from src.slices.Demo.TranscriptFormat import format_transcript

_STATIC = Path(__file__).parent / "static"
_EOF = "__eof__"  # client marks end-of-utterance so the server runs the final offline decode


def build_app(handler: StreamingDecoder_Handler) -> FastAPI:
    app = FastAPI(title="STREAM ASR demo")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.post("/transcribe")
    async def transcribe(file: UploadFile) -> JSONResponse:
        # Uploaded file (WAV/FLAC/OGG) -> full-context offline two-pass decode (best WER).
        raw = await file.read()
        try:
            wave = load_audio_bytes(raw)
        except Exception as exc:  # a non-audio / unsupported upload must not 500 the whole server
            return JSONResponse({"error": f"could not decode audio: {exc}"}, status_code=400)
        with torch.no_grad():
            resp = handler.decode_waveform(wave, streaming=False)
        seconds = wave.shape[0] / handler.cfg.audio.sample_rate
        return JSONResponse(
            {"text": format_transcript(resp.text), "rtf": resp.rtf, "seconds": seconds}
        )

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        # Binary frames = 16 kHz mono float32 PCM; a text `__eof__` frame ends the utterance.
        await ws.accept()
        session = StreamingSession(handler)
        try:
            while True:
                msg = await ws.receive()
                if (data := msg.get("bytes")) is not None:
                    pcm = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
                    partial = session.accept_audio(pcm)
                    await ws.send_json({"partial": format_transcript(partial)})
                elif msg.get("text") == _EOF:
                    resp = session.finalize()
                    await ws.send_json({"final": format_transcript(resp.text), "rtf": resp.rtf})
                    session.reset()  # ready for another utterance on the same socket
        except WebSocketDisconnect:
            return

    return app
