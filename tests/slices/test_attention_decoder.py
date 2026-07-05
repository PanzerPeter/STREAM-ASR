import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.AttentionDecoder import BiTransformerDecoder


def _inputs():
    cfg = get_config().model
    b, t, u = 2, 20, 7
    memory = torch.randn(b, t, cfg.encoder_dims[-1])
    memory_pad = torch.zeros(b, t, dtype=torch.bool)
    memory_pad[1, 15:] = True  # sample 1 is shorter
    ys_in = torch.randint(0, cfg.vocab_size, (b, u))
    ys_pad = torch.zeros(b, u, dtype=torch.bool)
    ys_pad[0, 5:] = True
    return cfg, memory, memory_pad, ys_in, ys_pad


def test_decoder_output_shape_both_directions():
    cfg, memory, memory_pad, ys_in, ys_pad = _inputs()
    dec = BiTransformerDecoder().eval()
    with torch.no_grad():
        left = dec(memory, memory_pad, ys_in, ys_pad, reverse=False)
        right = dec(memory, memory_pad, ys_in, ys_pad, reverse=True)
    assert left.shape == (2, 7, cfg.decoder_vocab_size)
    assert right.shape == (2, 7, cfg.decoder_vocab_size)


def test_decoder_self_attention_is_causal():
    # Changing a future target token must not change earlier-step logits (causal masking).
    cfg, memory, memory_pad, ys_in, ys_pad = _inputs()
    ys_pad = torch.zeros_like(ys_pad)  # no padding, isolate causality
    dec = BiTransformerDecoder().eval()
    with torch.no_grad():
        base = dec(memory, memory_pad, ys_in, ys_pad, reverse=False)
        bumped = ys_in.clone()
        bumped[:, -1] = (bumped[:, -1] + 1) % cfg.vocab_size
        after = dec(memory, memory_pad, bumped, ys_pad, reverse=False)
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-5)
