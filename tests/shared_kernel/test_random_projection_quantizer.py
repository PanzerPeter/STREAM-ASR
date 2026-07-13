import torch

from src.shared_kernel.RandomProjectionQuantizer import RandomProjectionQuantizer


def test_targets_deterministic_and_in_range():
    q = RandomProjectionQuantizer(in_dim=320, codebook_size=8192, codebook_dim=16, seed=7)
    x = torch.randn(2, 5, 320)
    t1 = q(x)
    t2 = q(x)
    assert t1.shape == (2, 5)
    assert torch.equal(t1, t2)  # deterministic across calls
    assert int(t1.min()) >= 0 and int(t1.max()) < 8192


def test_same_seed_same_codebook():
    a = RandomProjectionQuantizer(320, 8192, 16, seed=7)
    b = RandomProjectionQuantizer(320, 8192, 16, seed=7)
    assert torch.equal(a.codebook, b.codebook)
    assert torch.equal(a.proj, b.proj)


def test_parameters_are_frozen():
    q = RandomProjectionQuantizer(320, 8192, 16, seed=7)
    assert q.proj.requires_grad is False
    assert q.codebook.requires_grad is False
    assert len(list(q.parameters())) == 0  # buffers, not params
