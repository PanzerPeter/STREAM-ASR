# tests/slices/test_transducer_beam_search.py — TDD lock for TransducerBeamSearch (pure acoustic;
# the LM now rescores the n-best in StreamingDecoder_Handler, not per step): greedy must match the
# Task-8 trainer reference, and the time-synchronous A/B beam must return a well-formed, best-first,
# duplicate-free n-best. NOTE: beam_size=1 is deliberately NOT asserted equal to greedy any more --
# that identity only held for the earlier searcher that re-scored blank once per inner iteration
# (over-counting blank). The corrected search scores blank exactly once per frame and marginalises
# equal-prefix alignments via recombination, so its single-best is a true path score, not the
# locally-argmax greedy heuristic. greedy() itself is unchanged and still anchored to the trainer.
import math

import torch

from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import greedy_transducer_decode
from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch, _logadd, _recombine


class _Tok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def test_greedy_matches_trainer_reference() -> None:
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4, 5])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        ref = greedy_transducer_decode(model, memory, out_len, _Tok())[0]
        searcher = TransducerBeamSearch(model, beam_size=4, max_symbols=5)
        got_ids = searcher.greedy(memory[:, : int(out_len[0])])
    assert " ".join(str(i) for i in got_ids) == ref


def test_beam_size_one_returns_single_hypothesis() -> None:
    # beam=1 must still be a well-formed single-hypothesis search (the batched predictor+joiner
    # wiring runs with a 1-row batch). We no longer assert equality with greedy (see module note).
    torch.manual_seed(1)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        mem = memory[:, : int(out_len[0])]
        nbest = TransducerBeamSearch(model, beam_size=1, max_symbols=5).search(mem)
    assert len(nbest) == 1
    assert isinstance(nbest[0][1], float)


def test_batched_search_returns_best_first_nbest() -> None:
    # The batched beam must return float scores sorted best-first with at most beam_size hypotheses.
    torch.manual_seed(2)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4, 5])])
    beam_size = 4
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        mem = memory[:, : int(out_len[0])]
        nbest = TransducerBeamSearch(model, beam_size=beam_size, max_symbols=5).search(mem)
    assert 1 <= len(nbest) <= beam_size
    scores = [s for _, s in nbest]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)  # best-first


def test_nbest_has_no_duplicate_hypotheses() -> None:
    # Recombination must merge equal label sequences: the returned n-best carries no duplicate ids.
    torch.manual_seed(2)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(240, 80), [3, 4, 5, 6])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        mem = memory[:, : int(out_len[0])]
        nbest = TransducerBeamSearch(model, beam_size=8, max_symbols=5).search(mem)
    seqs = [tuple(ids) for ids, _ in nbest]
    assert len(seqs) == len(set(seqs))


def test_logadd_matches_reference() -> None:
    assert math.isclose(_logadd(-1.0, -1.0), math.log(2 * math.exp(-1.0)), rel_tol=1e-12)
    assert math.isclose(_logadd(-3.0, -100.0), -3.0, rel_tol=1e-9)  # small term negligible


def test_recombine_sums_equal_ids_via_logadd() -> None:
    # Two hyps with identical ids -> one entry whose score is the logadd of the two; distinct ids
    # pass through untouched. Output is best-first.
    st = torch.zeros(1, dtype=torch.long)
    hyps = [((7,), -1.0, st, 7), ((7,), -1.0, st, 7), ((8,), -0.5, st, 8)]
    out = _recombine(hyps)
    by_ids = {ids: score for ids, score, _, _ in out}
    assert len(out) == 2
    assert math.isclose(by_ids[(7,)], _logadd(-1.0, -1.0), rel_tol=1e-12)
    assert math.isclose(by_ids[(8,)], -0.5, rel_tol=1e-12)
    assert [s for _, s, _, _ in out] == sorted((s for _, s, _, _ in out), reverse=True)
