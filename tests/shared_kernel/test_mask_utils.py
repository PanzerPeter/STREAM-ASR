import torch

from src.shared_kernel.MaskUtils import make_chunk_mask, make_pad_mask


def test_pad_mask_marks_padding():
    lengths = torch.tensor([3, 5, 1])
    mask = make_pad_mask(lengths, max_len=5)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [False, False, False, True, True]
    assert mask[1].tolist() == [False] * 5
    assert mask[2].tolist() == [False, True, True, True, True]


def test_chunk_mask_full_context_when_nonpositive():
    assert make_chunk_mask(5, 0, -1, torch.device("cpu")) is None


def test_chunk_mask_is_chunk_causal_with_unlimited_left():
    # chunk_size=2, seq=5 -> chunks [0,1] [2,3] [4]
    m = make_chunk_mask(5, 2, -1, torch.device("cpu"))
    assert m is not None
    assert m[0].tolist() == [True, True, False, False, False]
    assert m[3].tolist() == [True, True, True, True, False]
    assert m[4].all()


def test_left_window_is_anchored_at_the_chunk_start():
    # Every query of one chunk must see the SAME history, or a streaming ring buffer could not
    # reproduce it: chunk [4, 6) with left 2 sees keys [2, 6) from both of its queries.
    m = make_chunk_mask(8, 2, 2, torch.device("cpu"))
    assert m is not None
    expected = [False, False, True, True, True, True, False, False]
    assert m[4].tolist() == expected and m[5].tolist() == expected
