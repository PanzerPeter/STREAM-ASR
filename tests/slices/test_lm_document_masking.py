import numpy as np
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.LmDataset import LmDataset
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


def test_segments_split_a_window_at_every_eos(tmp_path):
    # The packed corpus separates lines with EOS. A window's segment ids must increment AFTER each
    # EOS -- the EOS itself is the last token of the line it ends, not the first of the next.
    eos = get_config().model.eos_id
    bin_path = tmp_path / "toy.bin"
    np.asarray([7, 8, eos, 9, eos, 4, 5, 6], dtype=np.uint16).tofile(bin_path)
    x, _, seg = LmDataset(str(bin_path), context_len=6)[0]
    assert x.tolist() == [7, 8, eos, 9, eos, 4]
    assert seg.tolist() == [0, 0, 0, 1, 1, 2]


def test_masked_forward_equals_scoring_each_line_alone():
    # Document masking must make a packed window bit-equivalent to running each line separately:
    # that equivalence is the whole point -- training positions then see exactly the context a
    # rescored hypothesis sees at decode time.
    torch.manual_seed(0)
    lm = StreamLmModel().eval()
    eos = get_config().model.eos_id
    line_a, line_b = [7, 8, eos], [9, 4, 5]
    packed = torch.tensor([line_a + line_b])
    seg = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        joint = lm(packed, segments=seg)
        alone_a = lm(torch.tensor([line_a]))
        alone_b = lm(torch.tensor([line_b]))
    torch.testing.assert_close(joint[:, :3], alone_a, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(joint[:, 3:], alone_b, atol=2e-5, rtol=2e-5)


def test_unmasked_forward_still_sees_the_whole_window():
    # segments=None keeps the plain causal path (used by the single-sequence scorers), so the two
    # calls must differ -- otherwise the mask would be a no-op and the test above vacuous.
    torch.manual_seed(0)
    lm = StreamLmModel().eval()
    x = torch.tensor([[7, 8, get_config().model.eos_id, 9, 4, 5]])
    seg = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        assert not torch.allclose(lm(x), lm(x, segments=seg), atol=1e-4)
