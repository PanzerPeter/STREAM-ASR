import torch

from src.slices.PretrainEncoder.MelOnlyCollator import collate_mels
from src.shared_kernel.Config_Adapter import get_config


def test_collate_pads_and_reports_lengths():
    a = torch.randn(30, 80)
    b = torch.randn(20, 80)
    feats, lengths = collate_mels([a, b])
    assert feats.shape == (2, 30, 80)
    assert lengths.tolist() == [30, 20]
    assert torch.allclose(feats[1, :20], b)
    assert torch.count_nonzero(feats[1, 20:]) == 0  # padding is zero


def test_pretrain_config_loads():
    p = get_config().pretrain
    assert p.codebook_size > 0
    assert p.mask_span > 0
    assert p.stack_frames >= 0  # 0 = derive from encoder subsampling
