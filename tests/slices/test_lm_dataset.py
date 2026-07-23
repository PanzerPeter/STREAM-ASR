import numpy as np

from src.slices.TrainLanguageModel.LmDataset import LmDataset


def test_windows_shift_by_one(tmp_path):
    arr = np.arange(20, dtype=np.uint16)
    p = tmp_path / "toy.bin"
    arr.tofile(p)
    ds = LmDataset(str(p), context_len=4)
    assert len(ds) == 20 - 5  # windows of size context_len+1
    x, y, seg = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    assert seg.tolist() == [0, 0, 0, 0]  # no EOS in this toy stream -> one segment
    x2, _, _ = ds[3]
    assert x2.tolist() == [3, 4, 5, 6]
