import torch
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.TrainAcousticModel.StreamCache import StreamCache
from src.shared_kernel.Config_Adapter import get_config


def test_streaming_forward_equals_chunked_forward():
    torch.manual_seed(0)
    enc = ZipformerEncoder(cmvn_path=None).eval()
    n_mels = get_config().audio.n_mels
    base_chunk = enc.chunk_lcm() * 2  # B, base-rate frames per streaming step
    feat_chunk = base_chunk * 2  # feature-rate chunk fed to streaming_forward
    total = feat_chunk * 5
    x = torch.randn(1, total, n_mels)
    lengths = torch.tensor([total])
    with torch.no_grad():
        ref, _ = enc(x, lengths, chunk_size=base_chunk)  # non-streaming, same chunk-causal mask
        cache = StreamCache.init(enc, batch_size=1)
        outs = []
        for start in range(0, total, feat_chunk):
            y, cache = enc.streaming_forward(x[:, start : start + feat_chunk], cache)
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert stream.shape[1] == ref.shape[1]
    assert torch.allclose(ref, stream, atol=1e-4)  # all frames; causal encoder -> exact
