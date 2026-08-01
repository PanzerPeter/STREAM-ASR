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
    # Duck-types the members build_app touches, so the routes can be exercised without a 55M-param
    # forward: the assertions here are about transport + text presentation, not decoding.
    cfg = get_config()
    beam_size = 10
    lm_weight = 0.6
    ilm_weight = 0.2
    lm_scorer = object()  # truthy => /config reports the LM as attached
    model = torch.nn.Linear(1, 1)  # only reached for its parameters' device

    def decode_waveform(self, wave, streaming):
        return StreamingDecode_Response(
            text="I SAID HELLO",
            segments=[],
            rtf=0.1,
            decode_s=0.1,
            audio_s=1.0,
            finalize_s=0.01,
            first_partial_latency_s=0.0,
        )


def _endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


class _StubUpload:
    def __init__(self, raw):
        self._raw = raw

    async def read(self, size=-1):
        # Starlette's UploadFile.read takes a byte cap; the route uses it so an oversized upload is
        # never fully materialised.
        return self._raw if size < 0 else self._raw[:size]


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
    app = build_app(StreamingDecoder_Handler(model, _StubTok()), "transducer_avg.pt")
    paths = {getattr(r, "path", None) for r in app.routes}
    assert {"/", "/config", "/transcribe", "/stream"} <= paths


def test_transcribe_route_returns_sentence_cased_text():
    # The decoder emits upper-case corpus text; what leaves the server must be readable.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    app = build_app(_StubHandler(), "data/checkpoints/transducer_avg.pt")
    resp = asyncio.run(_endpoint(app, "/transcribe")(_StubUpload(buf.getvalue())))
    body = json.loads(resp.body)
    assert body["text"] == "I said hello"
    assert body["seconds"] == 1.0


def test_transcribe_route_rejects_non_audio_without_crashing():
    app = build_app(_StubHandler(), "data/checkpoints/transducer_avg.pt")
    resp = asyncio.run(_endpoint(app, "/transcribe")(_StubUpload(b"not audio at all")))
    assert resp.status_code == 400


def test_transcribe_route_rejects_oversized_upload():
    # The upload is held whole in memory before it can be decoded, so the size check has to happen
    # on the read itself, not after it.
    app = build_app(_StubHandler(), "data/checkpoints/transducer_avg.pt")
    resp = asyncio.run(_endpoint(app, "/transcribe")(_StubUpload(b"\0" * (65 * 1024 * 1024))))
    assert resp.status_code == 413


def test_config_route_reports_the_resolved_decode_settings():
    # What the page shows must be the settings the process actually runs with: a demo that fell back
    # to the alpha=0 regression lock, or to a non-averaged checkpoint, has to be visible in the UI.
    app = build_app(_StubHandler(), "data/checkpoints/transducer_avg.pt")
    body = json.loads(_endpoint(app, "/config")().body)
    assert body["checkpoint"] == "transducer_avg.pt"  # basename only, not the whole local path
    assert body["beam_size"] == 10
    assert body["lm"] == {"alpha": 0.6, "beta": 0.2}
    assert body["chunk_ms"] == round(1000 * get_config().decode.chunk_size / 50)


def test_config_route_reports_an_absent_lm_as_null():
    handler = _StubHandler()
    handler.lm_scorer = None
    app = build_app(handler, "data/checkpoints/transducer_best.pt")
    assert json.loads(_endpoint(app, "/config")().body)["lm"] is None
