import pytest
import torch

from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_dataloader_pin_memory_reaches_the_batch_tensors():
    # Guards the hot-path assumption behind .to(device, non_blocking=True): without a pin_memory
    # hook on the DTO, torch's walker returns the dataclass untouched and every H2D copy in
    # joint_loss silently becomes blocking again.
    batch = FeatureBatch(
        torch.randn(2, 4, 80),
        torch.tensor([4, 3]),
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([2, 2]),
    )
    pinned = torch.utils.data._utils.pin_memory.pin_memory(batch)
    assert isinstance(pinned, FeatureBatch)
    assert pinned.features.is_pinned()
    assert pinned.feature_lengths.is_pinned()
    assert pinned.tokens.is_pinned()
    assert pinned.token_lengths.is_pinned()
