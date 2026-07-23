# src/slices/Decode/LmScorer.py — decode-time wrapper over STREAM-LM for n-best rescoring: scores a
# whole hypothesis in one full-sequence forward. `sequence_score` applies the fusion weight (alpha)
# for the live decode path; `raw_sequence_logprob` returns the unweighted value so alpha tuning can
# sweep the weight over a fixed n-best without re-scoring. weight = 0 makes every score 0 -> the
# rescored ranking is identical to pure acoustic (the alpha=0 regression lock).
import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


class LmScorer:
    def __init__(self, model: StreamLmModel, weight: float) -> None:
        self.model = model
        self.weight = weight

    @torch.no_grad()
    def sequence_score(self, ids: list[int]) -> float:
        # Weighted sequence log-probability (sum of log-probs along the path) -- the term added to
        # the acoustic score when re-ranking the n-best at the configured alpha.
        return self.weight * self.model.sequence_logprob(ids)

    @torch.no_grad()
    def sequence_scores(self, nbest: list[list[int]]) -> list[float]:
        # Weighted scores for a whole n-best in ONE padded LM forward -- the live rescore path.
        # Per-hypothesis scoring made the LM ~half of offline decode time purely in launch
        # overhead (batch-1 forwards in a Python loop); the beam is small and uniform, so batching
        # it is free accuracy-wise and near-linear speed-wise.
        return [self.weight * lp for lp in self.model.sequence_logprob_batch(nbest)]

    @torch.no_grad()
    def raw_sequence_logprob(self, ids: list[int]) -> float:
        # Unweighted full-sequence log-probability -- the fusion weight (alpha) is applied by the
        # caller. Alpha tuning decodes dev once acoustic-only, then ranks a fixed n-best by
        # acoustic + alpha*this at every alpha with no further decoding, so the weight must not be
        # baked in here.
        return self.model.sequence_logprob(ids)

    @torch.no_grad()
    def raw_sequence_logprobs(self, nbest: list[list[int]]) -> list[float]:
        # Batched unweighted variant, used by alpha tuning to cache one n-best per utterance.
        return self.model.sequence_logprob_batch(nbest)
