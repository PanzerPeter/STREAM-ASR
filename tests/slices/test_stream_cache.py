from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.slices.TrainAcousticModel.StreamCache import StreamCache


def test_stream_cache_init_matches_encoder_layout():
    enc = ZipformerEncoder(cmvn_path=None)
    cache = StreamCache.init(enc, batch_size=1)
    total_blocks = sum(len(s.blocks) for s in enc.stacks)
    assert len(cache.attn) == total_blocks
    assert len(cache.conv) == total_blocks
    assert cache.frontend.in_tail.shape == (1, enc.frontend.time_pad, enc.cmvn_mean.shape[0])
    assert cache.frontend.mid_tail.shape[2] == enc.frontend.time_pad
    assert all(c.seen == 0 for c in cache.attn)
    assert all(c.k.shape[2] == 0 for c in cache.attn)
