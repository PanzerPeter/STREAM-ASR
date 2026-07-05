import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention


def _batch():
    torch.manual_seed(0)
    n_mels = get_config().audio.n_mels
    return collate_features(
        [(torch.randn(120, n_mels), [3, 4, 5, 6]), (torch.randn(90, n_mels), [7, 8, 9])]
    )


def test_forward_shapes_and_joint_loss():
    cfg = get_config()
    model = HybridCtcAttention(cmvn_path=None).train()
    batch = _batch()
    ctc_logits, memory, out_len = model(batch.features, batch.feature_lengths)
    assert ctc_logits.shape[-1] == cfg.model.logits_width  # 501
    assert memory.shape[-1] == cfg.model.encoder_dims[-1]  # 256
    total, ctc, attn = model.joint_loss(batch, chunk_size=0)
    assert torch.isfinite(total) and total.item() > 0
    # total == 0.3*ctc + 0.7*attn (config weights)
    sb = cfg.training.stage_b
    expected = sb.ctc_weight * ctc + (1 - sb.ctc_weight) * attn
    assert torch.allclose(total, expected, atol=1e-5)


def test_joint_loss_runs_chunked():
    model = HybridCtcAttention(cmvn_path=None).train()
    total, _, _ = model.joint_loss(_batch(), chunk_size=16)
    assert torch.isfinite(total)
