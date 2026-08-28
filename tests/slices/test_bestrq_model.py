import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.PretrainEncoder.BestRqModel import BestRqModel, stack_frames


def test_stack_frames_shape():
    x = torch.randn(2, 9, 4)
    s = stack_frames(x, 4)
    assert s.shape == (2, 2, 16)  # 9 // 4 = 2 target frames


def test_forward_returns_scalar_loss_and_accuracy():
    torch.manual_seed(0)
    model = BestRqModel(cmvn_path=None)
    mel = torch.randn(2, 128, 80)
    lengths = torch.tensor([128, 96])
    loss, acc = model(mel, lengths)
    assert loss.ndim == 0 and acc.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= acc.item() <= 1.0


def test_head_and_quantizers_cover_every_codebook():
    model = BestRqModel(cmvn_path=None)
    n = get_config().pretrain.num_codebooks
    k = get_config().pretrain.codebook_size
    assert len(model.quantizers) == n
    assert model.pred_head.out_features == n * k
    # Independent draws, or the mean of N cross-entropies is N copies of one.
    seeds = {q.codebook.sum().item() for q in model.quantizers}
    assert len(seeds) == n


def test_chunked_forward_runs_and_differs_from_full_context():
    # The stage samples chunk_size per batch, so limited right-context must reach the encoder here.
    torch.manual_seed(0)
    model = BestRqModel(cmvn_path=None).eval()
    mel = torch.randn(2, 256, 80)
    lengths = torch.tensor([256, 256])
    with torch.no_grad():
        torch.manual_seed(1)
        full, _ = model(mel, lengths, 0)
        torch.manual_seed(1)
        chunked, _ = model(mel, lengths, 16)
    assert torch.isfinite(chunked)
    assert not torch.isclose(full, chunked)


def test_one_step_reduces_loss_on_overfit_batch():
    torch.manual_seed(0)
    model = BestRqModel(cmvn_path=None)
    mel = torch.randn(2, 128, 80)
    lengths = torch.tensor([128, 128])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss, _ = model(mel, lengths)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
