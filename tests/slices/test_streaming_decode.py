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


def test_lm_weight_override_gates_and_threads(monkeypatch):
    # The --lm-weight override must (a) win over decode.lm_weight, and (b) gate LM loading: alpha>0
    # builds the scorer, alpha=0 keeps it None so the decoder stays byte-identical to the pre-LM
    # path (the alpha=0 regression lock). Checkpoint-free: _load_lm is stubbed to a sentinel.
    torch.manual_seed(0)
    model = HybridCtcAttention(cmvn_path=None).eval()
    loaded: list[float] = []

    def _fake_load_lm(self):
        loaded.append(self.lm_weight)
        return object()  # sentinel scorer; never invoked during construction

    monkeypatch.setattr(StreamingDecoder_Handler, "_load_lm", _fake_load_lm)

    on = StreamingDecoder_Handler(model, _StubTok(), fuse_lm_rescore=True, lm_weight=0.3)
    assert on.lm_weight == 0.3 and on.lm_scorer is not None and loaded == [0.3]

    off = StreamingDecoder_Handler(model, _StubTok(), fuse_lm_rescore=True, lm_weight=0.0)
    assert off.lm_scorer is None and loaded == [0.3]  # gate off -> _load_lm not called again
