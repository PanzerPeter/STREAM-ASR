import torch
from src.shared_kernel.MaskUtils import make_pad_mask, make_chunk_mask
from src.slices.TrainAcousticModel.ConvModule import ConvModule
from src.slices.TrainAcousticModel.RotaryAttention import RotaryAttention
from src.slices.TrainAcousticModel.StreamCache import AttnCache, ConvCache
from src.slices.TrainAcousticModel.ZipformerBlock import ZipformerBlock
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack


def test_conv_streaming_matches_full():
    torch.manual_seed(0)
    conv = ConvModule(32, kernel=15).eval()
    x = torch.randn(1, 30, 32)
    no_pad = make_pad_mask(torch.tensor([30]), 30)
    with torch.no_grad():
        full = conv(x, no_pad)
        left = torch.zeros(1, conv.kernel - 1, 32)
        outs = []
        for start in range(0, 30, 10):
            y, left = conv.streaming_forward(x[:, start : start + 10], left)
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(full, stream, atol=1e-5)


def test_attn_streaming_matches_chunk_forward():
    torch.manual_seed(0)
    attn = RotaryAttention(48, num_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 24, 48)
    pad = make_pad_mask(torch.tensor([24]), 24)
    chunk = 8
    visible = make_chunk_mask(24, chunk, x.device)  # chunk-causal reference
    with torch.no_grad():
        ref = attn(x, pad, visible)
        cache = AttnCache(torch.zeros(1, 4, 0, 12), torch.zeros(1, 4, 0, 12), 0)
        outs = []
        for start in range(0, 24, chunk):
            y, cache = attn.streaming_forward(x[:, start : start + chunk], cache)
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(ref, stream, atol=1e-5)


def test_block_streaming_matches_chunk_forward():
    torch.manual_seed(0)
    block = ZipformerBlock(48, num_heads=4).eval()
    x = torch.randn(1, 24, 48)
    pad = make_pad_mask(torch.tensor([24]), 24)
    chunk = 8
    visible = make_chunk_mask(24, chunk, x.device)
    with torch.no_grad():
        ref = block(x, pad, visible)
        ac = AttnCache(torch.zeros(1, 4, 0, 12), torch.zeros(1, 4, 0, 12), 0)
        cc = ConvCache(torch.zeros(1, block.conv.kernel - 1, 48))
        outs = []
        for start in range(0, 24, chunk):
            y, ac, cc = block.streaming_forward(x[:, start : start + chunk], ac, cc)
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(ref, stream, atol=1e-5)


def test_frontend_streaming_matches_full():
    from src.slices.TrainAcousticModel.Conv2dSubsampling import (
        Conv2dSubsampling,
    )
    from src.shared_kernel.Config_Adapter import get_config

    torch.manual_seed(0)
    front = Conv2dSubsampling().eval()
    n_mels = get_config().audio.n_mels
    x = torch.randn(1, 64, n_mels)
    lengths = torch.tensor([64])
    with torch.no_grad():
        full, _ = front(x, lengths)
        in_tail = torch.zeros(1, front.time_pad, n_mels)
        mid_tail = torch.zeros(1, front.conv1.out_channels, front.time_pad, front.freq_mid)
        outs = []
        for start in range(0, 64, 16):  # feature chunks of 16 -> 8 base frames each
            y, in_tail, mid_tail = front.streaming_forward(
                x[:, start : start + 16], in_tail, mid_tail
            )
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert stream.shape[1] == full.shape[1]
    assert torch.allclose(full, stream, atol=1e-5)


def test_stack_streaming_matches_chunk_forward():
    torch.manual_seed(0)
    stack = ZipformerStack(dim_in=48, dim=48, num_layers=2, downsample=4, num_heads=4).eval()
    x = torch.randn(1, 32, 48)  # 32 is a multiple of downsample 4
    lengths = torch.tensor([32])
    base_mask = make_pad_mask(lengths, 32)
    chunk = 8  # base-rate frames, multiple of downsample
    with torch.no_grad():
        ref = stack(x, lengths, base_mask, chunk_size=chunk)
        acs = [
            AttnCache(torch.zeros(1, 4, 0, 12), torch.zeros(1, 4, 0, 12), 0) for _ in stack.blocks
        ]
        ccs = [ConvCache(torch.zeros(1, b.conv.kernel - 1, 48)) for b in stack.blocks]
        outs = []
        for start in range(0, 32, chunk):
            y, acs, ccs = stack.streaming_forward(x[:, start : start + chunk], acs, ccs)
            outs.append(y)
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(ref, stream, atol=1e-5)
