# Decode-time wrapper over STREAM-LM for n-best rescoring. The LM has its own tokenizer, so it
# scores hypothesis TEXT: each hypothesis is re-encoded with the LM's SentencePiece model, which
# also makes the score independent of how the acoustic model split a word into BPE-500 pieces.
# `sequence_scores` applies the fusion weight (alpha) for the live decode path;
# `raw_sequence_logprobs` returns unweighted values so alpha tuning can sweep the weight over a
# fixed n-best without re-scoring. weight = 0 makes every score 0 -> the rescored ranking is
# identical to pure acoustic (the alpha=0 regression lock).
from typing import Protocol

import torch

from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


class LmScorer:
    def __init__(self, model: StreamLmModel, tokenizer: _Encoder, weight: float) -> None:
        self.model = model
        self.tok = tokenizer
        self.weight = weight

    @torch.no_grad()
    def sequence_scores(self, texts: list[str]) -> list[float]:
        # Weighted scores for a whole n-best in ONE padded LM forward -- the live rescore path.
        return [self.weight * lp for lp in self.raw_sequence_logprobs(texts)]

    @torch.no_grad()
    def raw_sequence_logprobs(self, texts: list[str]) -> list[float]:
        # Unweighted log-probabilities -- the fusion weight (alpha) is applied by the caller. Alpha
        # tuning decodes dev once acoustic-only, then ranks a fixed n-best by acoustic + alpha*this
        # at every alpha with no further decoding, so the weight must not be baked in here.
        return self.model.sequence_logprob_batch([self.tok.encode(t) for t in texts])
