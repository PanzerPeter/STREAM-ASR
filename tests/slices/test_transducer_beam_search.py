# TransducerBeamSearch is pure acoustic (the LM rescores its n-best in StreamingDecoder_Handler):
# the time-synchronous A/B beam must return a well-formed, best-first, duplicate-free n-best with
# one emission frame per token. Greedy is the model's label-looping search, locked in
# test_transducer_model.
import math

import torch

from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch, _logadd, _recombine
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


def _memory(seed: int, frames: int = 240) -> tuple[TransducerModel, torch.Tensor]:
    torch.manual_seed(seed)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(frames, 80), [3, 4, 5])])
    with torch.no_grad():
        memory, out_len, _ = model(b.features, b.feature_lengths)
    return model, memory[:, : int(out_len[0])]


def test_beam_size_one_returns_single_hypothesis() -> None:
    model, mem = _memory(1)
    with torch.no_grad():
        nbest = TransducerBeamSearch(model, beam_size=1, max_symbols=5).search(mem)
    assert len(nbest) == 1 and isinstance(nbest[0][1], float)


def test_search_returns_best_first_nbest_with_frames() -> None:
    model, mem = _memory(2)
    with torch.no_grad():
        nbest = TransducerBeamSearch(model, beam_size=4, max_symbols=5).search(mem)
    assert 1 <= len(nbest) <= 4
    scores = [s for _, s, _ in nbest]
    assert scores == sorted(scores, reverse=True)
    for ids, _, frames in nbest:
        assert len(frames) == len(ids) and frames == sorted(frames)
        assert all(0 <= f < mem.shape[1] for f in frames)


def test_nbest_has_no_duplicate_hypotheses() -> None:
    model, mem = _memory(2)
    with torch.no_grad():
        nbest = TransducerBeamSearch(model, beam_size=8, max_symbols=5).search(mem)
    seqs = [tuple(ids) for ids, _, _ in nbest]
    assert len(seqs) == len(set(seqs))


def test_logadd_matches_reference() -> None:
    assert math.isclose(_logadd(-1.0, -1.0), math.log(2 * math.exp(-1.0)), rel_tol=1e-12)
    assert math.isclose(_logadd(-3.0, -100.0), -3.0, rel_tol=1e-9)


def test_recombine_sums_equal_ids_and_keeps_the_likelier_frames() -> None:
    st = torch.zeros(1, dtype=torch.long)
    hyps = [((7,), -2.0, st, 7, (4,)), ((7,), -1.0, st, 7, (3,)), ((8,), -0.5, st, 8, (1,))]
    out = _recombine(hyps)
    by_ids = {h[0]: h for h in out}
    assert len(out) == 2
    assert math.isclose(by_ids[(7,)][1], _logadd(-2.0, -1.0), rel_tol=1e-12)
    assert by_ids[(7,)][4] == (3,)
    assert [h[1] for h in out] == sorted((h[1] for h in out), reverse=True)
