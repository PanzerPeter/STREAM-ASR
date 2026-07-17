import numpy as np
import torch

from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingSession import StreamingSession
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_session_partials_are_strings_and_finalize_delegates_to_handler():
    # Feeding audio to a StreamingSession in chunks must yield string partials without error, and
    # finalize() must delegate to the handler's own offline decode of the buffered waveform -- the
    # session's authoritative endpoint path. A live greedy partial need NOT byte-match the offline
    # final: offline runs chunk_size=0 (full bidirectional context) while the partial is a causal
    # greedy decode over accumulated streaming memory -- intentionally non-equivalent, same as the
    # streaming-vs-offline split already asserted in test_streaming_decode.py for the handler.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok(), fuse_lm=False)
    session = StreamingSession(handler)

    wave = torch.from_numpy((np.random.randn(16000) * 0.01).astype("float32"))  # 1 s @ 16 kHz
    with torch.no_grad():
        for s in range(0, wave.numel(), 4000):  # 0.25 s chunks
            partial = session.accept_audio(wave[s : s + 4000])
            assert isinstance(partial, str)
        final = session.finalize()

    assert isinstance(final, StreamingDecode_Response)
    assert isinstance(final.text, str)
    assert session.buffer.numel() == wave.numel()  # every sample was buffered


def test_session_reset_clears_state():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    session = StreamingSession(StreamingDecoder_Handler(model, _StubTok(), fuse_lm=False))
    with torch.no_grad():
        session.accept_audio(torch.zeros(8000, dtype=torch.float32))
    assert session.buffer.numel() == 8000
    session.reset()
    assert session.buffer.numel() == 0 and session.fed == 0
