import numpy as np
import soundfile as sf
import torch

from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _write_wav(tmp_path):
    sr = 16000
    wav = (np.random.randn(sr) * 0.01).astype("float32")  # 1 s
    p = tmp_path / "u.flac"
    sf.write(p, wav, sr)
    return str(p)


def test_streaming_and_offline_paths_run(tmp_path):
    torch.manual_seed(0)
    model = HybridCtcAttention(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok())
    path = _write_wav(tmp_path)
    with torch.no_grad():
        s = handler.decode(StreamingDecode_Command(audio_path=path, streaming=True))
        o = handler.decode(StreamingDecode_Command(audio_path=path, streaming=False))
    assert isinstance(s.text, str) and isinstance(o.text, str)
    assert s.rtf > 0 and s.first_partial_latency_s >= 0
    assert len(s.segments) >= 1
