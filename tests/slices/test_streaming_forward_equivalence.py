import pytest
import torch

from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def _stream(model: TransducerModel, feats: torch.Tensor, chunk: int, left: int, tail: int):
    enc = model.encoder
    mel_chunk = chunk * 4
    n = feats.shape[1]
    n_chunks = -(-n // mel_chunk) + tail
    x = torch.cat([feats, torch.zeros(1, n_chunks * mel_chunk - n, feats.shape[2])], dim=1)
    cache = enc.init_stream(1, left)
    outs = []
    for i in range(n_chunks):
        s = i * mel_chunk
        y, cache = enc.streaming_forward(
            x[:, s : s + mel_chunk], cache, max(0, min(mel_chunk, n - s))
        )
        outs.append(y)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize(
    "n_mel,chunk,left",
    [(480, 8, 64), (437, 8, 16), (437, 16, -1), (301, 4, 4)],
)
def test_streaming_forward_equals_batched_forward(n_mel, chunk, left):
    # Invariant 3: streaming_forward == forward under the same (chunk, left) mask to fp32
    # round-off, including a partial final chunk (mean fill, key-masked) and a kept tail chunk.
    # Window-local RoPE changes the angles' rounding, not the function, hence a tolerance.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    feats = torch.randn(1, n_mel, 80)
    with torch.no_grad():
        ref, keep = model.decode_window(feats, torch.tensor([n_mel]), chunk, left, 1)
        stream = _stream(model, feats, chunk, left, 1)
    k = int(keep)
    assert stream.shape[1] == k
    assert torch.allclose(ref[:, :k], stream, atol=1e-4)


def test_stream_cache_stays_bounded():
    # A bounded left context makes the attention state constant-size for any stream length.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    enc = model.encoder
    cache = enc.init_stream(1, 16)
    with torch.no_grad():
        for _ in range(12):
            _, cache = enc.streaming_forward(torch.randn(1, 32, 80), cache)
    lengths = {b.k.shape[2] for b in cache.blocks}
    assert lengths == {16, 16 // 2}  # S0 at 25 Hz keeps 16 frames, the 12.5 Hz stacks 8


def test_unaligned_window_is_rejected():
    model = TransducerModel(cmvn_path=None)
    with pytest.raises(ValueError, match="multiples of 4"):
        model.encoder(torch.randn(1, 64, 80), torch.tensor([64]), chunk_size=6)
