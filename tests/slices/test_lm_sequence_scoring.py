import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


def _lm() -> StreamLmModel:
    torch.manual_seed(0)
    lm = StreamLmModel().eval()
    nn.init.normal_(lm.head.weight)  # the zero-init readout would make every score -len*ln(V)
    return lm


def test_sequence_logprob_scores_eos_plus_line():
    # A hypothesis is scored as the row LmDataset trains every line as: input [EOS] + ids,
    # targets ids + [EOS].
    lm = _lm()
    eos = get_config().lm.eos_id
    ids = [3, 7, 42]
    with torch.no_grad():
        logp = torch.log_softmax(lm(torch.tensor([[eos] + ids]))[0], dim=-1)
    target = torch.tensor(ids + [eos])
    expected = float(logp[torch.arange(target.shape[0]), target].sum())
    assert abs(lm.sequence_logprob(ids) - expected) < 1e-4


def test_batched_sequence_logprob_matches_per_sequence_scoring():
    # The n-best rescorer scores a whole beam in one padded forward; padding must not leak into a
    # shorter hypothesis' score, so each row has to equal its own single-sequence score.
    lm = _lm()
    seqs = [[5], [3, 7, 42, 100], [11, 12], []]
    batched = lm.sequence_logprob_batch(seqs)
    singles = [lm.sequence_logprob(s) for s in seqs]
    assert len(batched) == len(seqs)
    for b, s in zip(batched, singles):
        assert abs(b - s) < 1e-3


def test_batched_sequence_logprob_on_empty_input():
    assert _lm().sequence_logprob_batch([]) == []
