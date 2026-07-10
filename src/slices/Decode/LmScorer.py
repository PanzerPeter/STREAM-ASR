# src/slices/Decode/LmScorer.py — Thin decode-time wrapper over STREAM-LM: applies the
# fusion weight (alpha) to both the incremental next-token log-probs (first-pass shallow fusion)
# and the full-sequence log-prob (second-pass rescore). weight = 0 makes every score 0 -> decode
# is unchanged.
import torch

from src.slices.TrainLanguageModel.CausalGqaAttention import KvCache
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel


class LmScorer:
    def __init__(self, model: StreamLmModel, weight: float) -> None:
        self.model = model
        self.weight = weight

    @torch.no_grad()
    def step_score(
        self, token: int, state: list[KvCache] | None
    ) -> tuple[torch.Tensor, list[KvCache]]:
        # Returns the weighted log-probability tensor and the next LM state. Inference-only: no_grad
        # keeps decode from building an autograd graph over the LM.
        logp, state = self.model.step_logprob(token, state)
        # weight 0 must be an exact no-op even if logp holds -inf (0 * -inf = nan otherwise).
        if self.weight == 0.0:
            return torch.zeros_like(logp), state
        return self.weight * logp, state

    @torch.no_grad()
    def sequence_score(self, ids: list[int]) -> float:
        # Returns the weighted sequence log-probability (sum of log-probs along the path).
        return self.weight * self.model.sequence_logprob(ids)
