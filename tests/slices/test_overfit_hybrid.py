import pytest
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention


@pytest.mark.slow
def test_hybrid_overfits_one_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_mels = get_config().audio.n_mels
    torch.manual_seed(0)
    batch = collate_features(
        [(torch.randn(140, n_mels), [3, 4, 5, 6, 7]), (torch.randn(110, n_mels), [8, 9, 10, 11])]
    )
    model = HybridCtcAttention(cmvn_path=None).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        total, _, _ = model.joint_loss(batch, chunk_size=0)
        total.backward()
        opt.step()
        first = first if first is not None else total.item()
    assert total.item() < 0.5 * first  # joint loss more than halves on a single batch
