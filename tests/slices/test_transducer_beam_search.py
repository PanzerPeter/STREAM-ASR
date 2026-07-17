# tests/slices/test_transducer_beam_search.py — TDD lock for TransducerBeamSearch (pure acoustic;
# the LM now rescores the n-best in StreamingDecoder_Handler, not per step): greedy must match the
# Task-8 trainer reference, beam_size=1 must degenerate to greedy (validates state threading through
# repeated predictor.step calls), and the batched beam must return a well-formed, best-first n-best.
import torch

from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.TrainAcousticModel.TransducerTrainer_Handler import greedy_transducer_decode
from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch


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


def test_beam_size_one_equals_greedy() -> None:
    # With beam=1 the batched search reduces to a single-row batch, so this is the exact (argmax,
    # fp-robust) anchor that the batched predictor+joiner wiring is correct.
    torch.manual_seed(1)
    model = TransducerModel(cmvn_path=None).eval()
    b = collate_features([(torch.randn(160, 80), [3, 4])])
    with torch.no_grad():
        memory, out_len, _, _, _ = model(b.features, b.feature_lengths)
        mem = memory[:, : int(out_len[0])]
        searcher = TransducerBeamSearch(model, beam_size=1, max_symbols=5)
        nbest = searcher.search(mem)
        greedy = searcher.greedy(mem)
    assert nbest[0][0] == greedy


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
