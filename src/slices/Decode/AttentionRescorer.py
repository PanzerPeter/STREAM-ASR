# Second pass: score each first-pass hypothesis with the bidirectional decoder and blend with
# the CTC score. Teacher-forces the whole hypothesis in one forward per direction (rescoring,
# not search).
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config


class _SeqScorer(Protocol):
    # Structural type: the rescorer only needs the LM's weighted full-sequence log-prob, not the
    # concrete LmScorer class (mirrors why `decoder` is typed as nn.Module).
    def sequence_score(self, ids: list[int]) -> float: ...


class AttentionRescorer:
    # Typed as nn.Module (not the concrete BiTransformerDecoder) so the Decode slice does not
    # import a training-internal class just for an annotation; it only needs the callable contract.
    def __init__(self, decoder: nn.Module, lm_scorer: _SeqScorer | None = None) -> None:
        self.decoder = decoder
        self.lm = lm_scorer
        m = get_config().model
        self.sos, self.eos = m.sos_id, m.eos_id
        self.lam = get_config().decode.rescore_lambda

    def _seq_logprob(
        self, memory: torch.Tensor, mem_pad: torch.Tensor, hyp: list[int], reverse: bool
    ) -> float:
        # Compute the sum of log probabilities of the hypothesis under the decoder,
        # accounting for direction (L2R or R2L via reversed).
        toks = list(reversed(hyp)) if reverse else hyp
        ys_in = torch.tensor([[self.sos] + toks], device=memory.device)
        ys_out = torch.tensor([toks + [self.eos]], device=memory.device)
        ys_pad = torch.zeros_like(ys_in, dtype=torch.bool)
        logits = self.decoder(memory, mem_pad, ys_in, ys_pad, reverse=reverse)
        logp = F.log_softmax(logits, dim=-1)
        return float(logp[0, torch.arange(ys_out.shape[1]), ys_out[0]].sum())

    def rescore(
        self,
        memory: torch.Tensor,
        mem_pad: torch.Tensor,
        nbest: list[tuple[int, ...]],
        ctc_scores: list[float],
    ) -> list[tuple[list[int], float]]:
        # Rescore each hypothesis with the bidirectional decoder. Empty hypotheses keep their
        # CTC score unchanged. Score blends L2R and R2L log probabilities with rescore_lambda.
        results = []
        for hyp, ctc in zip(nbest, ctc_scores):
            hyp_list = list(hyp)
            if not hyp_list:
                results.append((hyp_list, ctc))
                continue
            l2r = self._seq_logprob(memory, mem_pad, hyp_list, reverse=False)
            r2l = self._seq_logprob(memory, mem_pad, hyp_list, reverse=True)
            # LM rescore term: 0 when no scorer, so the blend is unchanged at alpha=0.
            lm_term = self.lm.sequence_score(hyp_list) if self.lm is not None else 0.0
            results.append((hyp_list, ctc + (1 - self.lam) * l2r + self.lam * r2l + lm_term))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
