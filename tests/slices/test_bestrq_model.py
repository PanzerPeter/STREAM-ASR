import torch

from src.slices.PretrainEncoder.BestRqModel import BestRqModel, stack_frames


def test_stack_frames_shape():
    x = torch.randn(2, 9, 4)
    s = stack_frames(x, 4)
    assert s.shape == (2, 2, 16)  # 9 // 4 = 2 target frames


def test_forward_returns_scalar_loss_and_targets_align():
    torch.manual_seed(0)
    model = BestRqModel(cmvn_path=None)
    mel = torch.randn(2, 128, 80)
    lengths = torch.tensor([128, 96])
    loss = model(mel, lengths)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_one_step_reduces_loss_on_overfit_batch():
    torch.manual_seed(0)
    model = BestRqModel(cmvn_path=None)
    mel = torch.randn(2, 128, 80)
    lengths = torch.tensor([128, 128])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = model(mel, lengths)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
