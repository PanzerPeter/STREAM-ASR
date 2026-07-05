import torch
from src.slices.TrainAcousticModel.CtcGreedyDecoder import ctc_greedy_decode
from src.shared_kernel.Config_Adapter import get_config


class _FakeTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)  # identity-ish decode for assertion


def test_greedy_collapses_repeats_and_drops_blank():
    model = get_config().model
    V = model.logits_width
    # Build an argmax path [5, 5, blank, 7, 7] -> collapse repeats, drop blank -> [5, 7].
    path = [5, 5, model.blank_id, 7, 7]
    logits = torch.full((1, len(path), V), -10.0)
    for t, idx in enumerate(path):
        logits[0, t, idx] = 10.0
    out = ctc_greedy_decode(logits, torch.tensor([len(path)]), _FakeTok())
    assert out == ["5 7"]
