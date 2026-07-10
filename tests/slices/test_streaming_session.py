import numpy as np
import torch

from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingSession import StreamingSession
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_session_partials_and_final_match_offline_decode():
    # Feeding audio to a StreamingSession in chunks must (a) yield string partials without error and
    # (b) finalize to exactly the handler's offline decode of the same buffered waveform — the
    # session's authoritative endpoint path. Random init + fixed seed keeps it checkpoint-free.
    torch.manual_seed(0)
    model = HybridCtcAttention(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok())
    session = StreamingSession(handler)

    wave = torch.from_numpy((np.random.randn(16000) * 0.01).astype("float32"))  # 1 s @ 16 kHz
    with torch.no_grad():
        for s in range(0, wave.numel(), 4000):  # 0.25 s chunks
            partial = session.accept_audio(wave[s : s + 4000])
            assert isinstance(partial, str)
        final = session.finalize()
        reference = handler.decode_waveform(wave, streaming=False)

    assert final.text == reference.text
    assert session.buffer.numel() == wave.numel()  # every sample was buffered


def test_session_reset_clears_state():
    torch.manual_seed(0)
    model = HybridCtcAttention(cmvn_path=None).eval()
    session = StreamingSession(StreamingDecoder_Handler(model, _StubTok()))
    with torch.no_grad():
        session.accept_audio(torch.zeros(8000, dtype=torch.float32))
    assert session.buffer.numel() == 8000
    session.reset()
    assert session.buffer.numel() == 0 and session.fed == 0
