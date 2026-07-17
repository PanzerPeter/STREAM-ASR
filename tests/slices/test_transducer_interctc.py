import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def test_interctc_head_widths_match_tapped_stacks():
    model = TransducerModel(cmvn_path=None)
    dims = get_config().model.encoder_dims
    layers = get_config().transducer.interctc_layers
    for head, idx in zip(model.interctc_heads, layers):
        assert head.in_features == dims[idx]
        assert head.out_features == get_config().model.logits_width


def test_interctc_loss_is_finite_and_positive():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).train()
    b = collate_features([(torch.randn(160, 80), [3, 4, 5, 6, 7])])
    _, _, _, interctc, base_len = model(b.features, b.feature_lengths)
    loss = model.interctc_loss(interctc, base_len, b.tokens, b.token_lengths)
    assert torch.isfinite(loss) and loss > 0
