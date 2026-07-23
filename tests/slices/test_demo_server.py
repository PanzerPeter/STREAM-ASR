import asyncio
import io
import json

import numpy as np
import soundfile as sf
import torch

from src.shared_kernel.AudioIO_Adapter import load_audio_bytes
from src.shared_kernel.Config_Adapter import get_config
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Demo.DemoServer_Handler import build_app
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _StubHandler:
    # Duck-types the two members build_app touches, so the route can be exercised without a
    # 55M-param forward: the assertion here is about transport + text presentation, not decoding.
    cfg = get_config()

    def decode_waveform(self, wave, streaming):
        return StreamingDecode_Response(
            text="I SAID HELLO", segments=[], rtf=0.1, first_partial_latency_s=0.0
        )


def _endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


class _StubUpload:
    def __init__(self, raw):
        self._raw = raw

    async def read(self):
        return self._raw


def test_load_audio_bytes_decodes_and_resamples():
    # An in-memory 8 kHz stereo WAV must decode to mono at the config rate (16 kHz): downmix +
    # resample, matching load_audio's path but from bytes (the upload route's entry).
    buf = io.BytesIO()
    sf.write(buf, np.zeros((8000, 2), dtype="float32"), 8000, format="WAV")  # 1 s stereo @ 8 kHz
    wave = load_audio_bytes(buf.getvalue())
    assert wave.ndim == 1 and wave.dtype == torch.float32
    assert abs(wave.numel() - 16000) <= 2  # resampled 8 kHz -> 16 kHz


def test_build_app_registers_routes():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    app = build_app(StreamingDecoder_Handler(model, _StubTok()))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert {"/", "/transcribe", "/stream"} <= paths


def test_transcribe_route_returns_sentence_cased_text():
    # The decoder emits upper-case corpus text; what leaves the server must be readable.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    app = build_app(_StubHandler())
    resp = asyncio.run(_endpoint(app, "/transcribe")(_StubUpload(buf.getvalue())))
    body = json.loads(resp.body)
    assert body["text"] == "I said hello"
    assert body["seconds"] == 1.0


def test_transcribe_route_rejects_non_audio_without_crashing():
    app = build_app(_StubHandler())
    resp = asyncio.run(_endpoint(app, "/transcribe")(_StubUpload(b"not audio at all")))
    assert resp.status_code == 400
