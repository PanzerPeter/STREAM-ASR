import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


def _lm() -> StreamLmModel:
    torch.manual_seed(0)
    return StreamLmModel().eval()


def test_sequence_logprob_starts_from_bos_which_is_eos():
    # The start symbol must be the id the corpus actually trains ("previous line's EOS"), not a
    # never-seen start row -- otherwise every hypothesis' first-token score is untrained noise.
    lm = _lm()
    m = get_config().model
    ids = [3, 7, 42]
    seq = torch.tensor([[m.bos_id] + ids])
    with torch.no_grad():
        logp = torch.log_softmax(lm(seq)[0], dim=-1)
    target = torch.tensor(ids + [m.eos_id])
    expected = float(logp[torch.arange(target.shape[0]), target].sum())
    assert m.bos_id == m.eos_id
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
