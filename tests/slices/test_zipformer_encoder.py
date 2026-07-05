import torch
from src.slices.TrainAcousticModel.ZipformerEncoder import ZipformerEncoder
from src.shared_kernel.Config_Adapter import get_config


def test_encoder_output_rate_and_dim():
    cfg = get_config()
    enc = ZipformerEncoder(cmvn_path=None).eval()
    features = torch.randn(2, 200, cfg.audio.n_mels)
    lengths = torch.tensor([200, 150])
    with torch.no_grad():
        out, out_len = enc(features, lengths)
    assert out.shape[0] == 2
    assert out.shape[2] == cfg.model.encoder_dims[-1]  # 256
    # ~×4 subsampling: 200 -> ~50 frames (±1 for conv/ceil rounding).
    assert abs(out.shape[1] - 200 // 4) <= 2
    assert (out_len <= out.shape[1]).all()
    assert enc.output_dim == cfg.model.encoder_dims[-1]


def test_encoder_param_count_in_range():
    enc = ZipformerEncoder(cmvn_path=None)
    millions = sum(p.numel() for p in enc.parameters()) / 1e6
    assert 40 <= millions <= 110, f"{millions:.1f}M params outside sane band"


def test_encoder_chunked_matches_shape_and_differs_from_full():
    cfg = get_config()
    torch.manual_seed(0)
    enc = ZipformerEncoder(cmvn_path=None).eval()
    features = torch.randn(2, 200, cfg.audio.n_mels)
    lengths = torch.tensor([200, 200])
    with torch.no_grad():
        full, len_full = enc(features, lengths)  # chunk_size defaults to 0 = full context
        chunked, len_chunked = enc(features, lengths, chunk_size=16)
    assert chunked.shape == full.shape
    assert torch.equal(len_chunked, len_full)  # masking never changes lengths
    # Restricting attention context must change the output (proves the mask is actually applied).
    assert not torch.allclose(chunked, full, atol=1e-4)


def test_encoder_grad_checkpoint_path_runs_full_and_chunked():
    # The activation-checkpointing wrapper must accept the chunk_size arg the encoder now passes,
    # at both the default full-context and a chunked setting (regression guard).
    import torch.nn as nn
    from src.slices.TrainAcousticModel.StageATrainer_Handler import _Checkpointed

    cfg = get_config()
    enc = ZipformerEncoder(cmvn_path=None).eval()
    enc.stacks = nn.ModuleList([_Checkpointed(s) for s in enc.stacks])
    features = torch.randn(2, 160, cfg.audio.n_mels, requires_grad=True)
    lengths = torch.tensor([160, 120])
    for cs in (0, 16):
        out, out_len = enc(features, lengths, chunk_size=cs)
        assert out.shape[0] == 2 and out.shape[2] == cfg.model.encoder_dims[-1]
