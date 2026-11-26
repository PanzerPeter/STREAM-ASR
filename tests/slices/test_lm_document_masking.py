import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


def _lm() -> StreamLmModel:
    # The readout is zero-initialised (every logit 0), which would make any equality below
    # vacuous; a random readout makes the logits depend on the whole stack.
    torch.manual_seed(0)
    lm = StreamLmModel().eval()
    nn.init.normal_(lm.head.weight)
    return lm


def test_masked_forward_equals_scoring_each_line_from_eos():
    # Document masking must make a packed window equal to scoring each line on its own as
    # [EOS] + line, which is the exact row sequence_logprob_batch builds at decode time.
    lm = _lm()
    eos = get_config().lm.eos_id
    line_a, line_b = [eos, 7, 8], [eos, 9, 4, 5]
    packed = torch.tensor([line_a + line_b])
    seg = (packed == eos).long().cumsum(1)  # LmDataset's segment rule
    with torch.no_grad():
        joint = lm(packed, segments=seg)
        alone_a = lm(torch.tensor([line_a]))
        alone_b = lm(torch.tensor([line_b]))
    # atol at 1e-4: exact-equivalence float error scales with d_model (512 accumulates ~4e-5), well
    # below the O(1) gap a broken document mask would open, so the check stays meaningful.
    torch.testing.assert_close(joint[:, :3], alone_a, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(joint[:, 3:], alone_b, atol=1e-4, rtol=1e-4)


def test_unmasked_forward_still_sees_the_whole_window():
    # segments=None keeps the plain causal path (used by the single-sequence scorers), so the two
    # calls must differ -- otherwise the mask would be a no-op and the test above vacuous.
    lm = _lm()
    eos = get_config().lm.eos_id
    x = torch.tensor([[eos, 7, 8, eos, 9, 4]])
    seg = (x == eos).long().cumsum(1)
    with torch.no_grad():
        assert not torch.allclose(lm(x), lm(x, segments=seg), atol=1e-4)
