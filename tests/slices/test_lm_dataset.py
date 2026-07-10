import numpy as np

from src.slices.TrainLanguageModel.LmDataset import LmDataset


def test_windows_shift_by_one(tmp_path):
    arr = np.arange(20, dtype=np.uint16)
    p = tmp_path / "toy.bin"
    arr.tofile(p)
    ds = LmDataset(str(p), context_len=4)
    assert len(ds) == 20 - 5  # windows of size context_len+1
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    x2, y2 = ds[3]
    assert x2.tolist() == [3, 4, 5, 6]
