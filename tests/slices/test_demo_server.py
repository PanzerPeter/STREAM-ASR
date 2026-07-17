import io

import numpy as np
import soundfile as sf
import torch

from src.shared_kernel.AudioIO_Adapter import load_audio_bytes
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Demo.DemoServer_Handler import build_app
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


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
