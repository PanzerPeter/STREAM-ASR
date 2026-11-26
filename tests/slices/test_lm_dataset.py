import numpy as np
import pytest

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.LmDataset import LmDataset


def test_windows_tile_the_stream_without_overlap(tmp_path):
    # One pass over the dataset must read every token once: window i starts at i * context_len.
    arr = np.arange(2, 22, dtype=np.uint16)  # 20 tokens, none of them the EOS id
    p = tmp_path / "toy.bin"
    arr.tofile(p)
    ds = LmDataset(str(p), context_len=4)
    assert len(ds) == (20 - 1) // 4
    x, y, seg = ds[0]
    assert x.tolist() == [2, 3, 4, 5]
    assert y.tolist() == [3, 4, 5, 6]
    assert seg.tolist() == [0, 0, 0, 0]  # no EOS in this toy stream -> one segment
    x2, y2, _ = ds[3]
    assert x2.tolist() == [14, 15, 16, 17] and y2.tolist() == [15, 16, 17, 18]
    starts = [int(ds[i][0][0]) for i in range(len(ds))]
    assert starts == [2, 6, 10, 14]


def test_segment_ids_advance_on_each_eos(tmp_path):
    # The EOS that ends a line opens the next one, so every packed line trains as [EOS] + line --
    # the row the rescorer scores. Its target (the next line's first token) is predicted from the
    # EOS alone, exactly as at decode.
    eos = get_config().lm.eos_id
    p = tmp_path / "toy.bin"
    np.asarray([7, 8, eos, 9, eos, 4, 5, 6], dtype=np.uint16).tofile(p)
    x, y, seg = LmDataset(str(p), context_len=6)[0]
    assert x.tolist() == [7, 8, eos, 9, eos, 4]
    assert y.tolist() == [8, eos, 9, eos, 4, 5]
    assert seg.tolist() == [0, 0, 1, 1, 2, 2]


def test_a_bin_shorter_than_one_window_is_rejected(tmp_path):
    p = tmp_path / "toy.bin"
    np.arange(2, 6, dtype=np.uint16).tofile(p)
    with pytest.raises(ValueError):
        LmDataset(str(p), context_len=4)
