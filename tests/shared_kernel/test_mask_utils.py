import torch
from src.shared_kernel.MaskUtils import make_pad_mask, make_chunk_mask


def test_pad_mask_marks_padding():
    lengths = torch.tensor([3, 5, 1])
    mask = make_pad_mask(lengths, max_len=5)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [False, False, False, True, True]
    assert mask[1].tolist() == [False] * 5
    assert mask[2].tolist() == [False, True, True, True, True]


def test_chunk_mask_full_context_when_nonpositive():
    m = make_chunk_mask(5, 0, torch.device("cpu"))
    assert m.shape == (5, 5)
    assert m.all()  # full context: everything visible


def test_chunk_mask_is_chunk_causal():
    # chunk_size=2, seq=5 -> chunks [0,1] [2,3] [4]
    m = make_chunk_mask(5, 2, torch.device("cpu"))
    # frame 0 (chunk 0) sees chunk 0 only: cols 0,1 True; 2,3,4 False
    assert m[0].tolist() == [True, True, False, False, False]
    # frame 3 (chunk 1) sees chunks 0 and 1: cols 0..3 True; 4 False
    assert m[3].tolist() == [True, True, True, True, False]
    # frame 4 (chunk 2) sees everything
    assert m[4].all()
    # a later chunk is never visible to an earlier one (upper-right block is False)
    assert not m[0, 2]
