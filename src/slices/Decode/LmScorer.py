# Decode-time wrapper over STREAM-LM for n-best rescoring: scores a whole n-best in one padded
# forward. `sequence_scores` applies the fusion weight (alpha) for the live decode path;
# `raw_sequence_logprobs` returns unweighted values so alpha tuning can sweep the weight over a
# fixed n-best without re-scoring. weight = 0 makes every score 0 -> the rescored ranking is
# identical to pure acoustic (the alpha=0 regression lock).
import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


class LmScorer:
    def __init__(self, model: StreamLmModel, weight: float) -> None:
        self.model = model
        self.weight = weight

    @torch.no_grad()
    def sequence_scores(self, nbest: list[list[int]]) -> list[float]:
        # Weighted scores for a whole n-best in ONE padded LM forward -- the live rescore path.
        # Scoring hypotheses one at a time costs a batch-1 forward each and puts the LM at ~half of
        # offline decode time in launch overhead alone; the beam is small and uniform, so batching
        # is free accuracy-wise and near-linear in speed.
        return [self.weight * lp for lp in self.model.sequence_logprob_batch(nbest)]

    @torch.no_grad()
    def raw_sequence_logprobs(self, nbest: list[list[int]]) -> list[float]:
        # Unweighted log-probabilities -- the fusion weight (alpha) is applied by the caller. Alpha
        # tuning decodes dev once acoustic-only, then ranks a fixed n-best by acoustic + alpha*this
        # at every alpha with no further decoding, so the weight must not be baked in here.
        return self.model.sequence_logprob_batch(nbest)
