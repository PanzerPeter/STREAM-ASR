# src/slices/Decode/InternalLmScorer.py — internal-LM estimation (ILME) for the RNN-T.
#
# An RNN-T's score already contains a language prior: the predictor+joiner learned one from the
# 960 h transcripts. Adding an external LM on top therefore counts language evidence twice, which
# caps how much the external LM can buy. ILME removes the double count by subtracting an estimate
# of that internal prior at re-ranking time:
#
#     score = acoustic + alpha * lm_external - beta * lm_internal
#
# The estimate follows Meng et al. (arXiv:2011.01991): evaluate the joiner with the ACOUSTIC
# CONTRIBUTION ZEROED (encoder memory set to the zero vector, so only the joiner's own bias and the
# predictor path survive) and renormalise over the non-blank labels. With this repo's stateless
# predictor the resulting prior is inherently low-order -- it conditions on `predictor_context`
# tokens -- which is exactly the regime LODR argues is the right thing to subtract.
#
# The estimate has no EOS symbol (the transducer's label space has none), so an ILM sequence score
# covers the U emitted tokens only; the external LM's EOS term is left to act as its usual mild
# length prior.
import torch
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class InternalLmScorer:
    def __init__(self, model: TransducerModel) -> None:
        m = get_config().model
        self.model = model
        self.blank = m.blank_id
        if self.blank != m.logits_width - 1:
            # The non-blank renormalisation below slices the blank off the END of the logits.
            raise ValueError(f"blank_id {self.blank} must be the last of {m.logits_width} logits")

    @torch.no_grad()
    def sequence_logprob(self, ids: list[int]) -> float:
        return self.sequence_logprob_batch([list(ids)])[0]

    @torch.no_grad()
    def sequence_logprob_batch(self, seqs: list[list[int]]) -> list[float]:
        # One padded predictor+joiner pass for a whole n-best, mirroring the external LM's batched
        # scorer: the predictor is causal over its own left context, so padding a shorter
        # hypothesis on the right cannot change its earlier positions, and the per-row sum masks
        # the padding out.
        if not seqs:
            return []
        device = self.model.ctc_head.weight.device
        lengths = [len(s) for s in seqs]
        width = max(lengths)
        if width == 0:
            return [0.0] * len(seqs)
        # Prediction inputs are blank-prefixed (the same convention as TransducerModel.rnnt_loss),
        # so position u is the predictor state having consumed ids[:u] -- the state that predicts
        # ids[u]. Only the first `width` positions are needed; the final state predicts nothing.
        # Padding value is arbitrary (0): the predictor only reads leftwards, so pad positions
        # cannot alter a shorter hypothesis' real positions, and the row sum masks them out.
        labels = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if s:
                labels[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        blanks = torch.full((len(seqs), 1), self.blank, dtype=torch.long, device=device)
        pred = self.model.predictor(torch.cat([blanks, labels], dim=1)[:, :width])
        zero_enc = torch.zeros(
            len(seqs), width, self.model.encoder.output_dim, device=device, dtype=pred.dtype
        )
        # joiner.step broadcasts elementwise over the [K, width] grid, giving the diagonal cells
        # (t == u) the lattice never needs here -- there is no acoustic axis left to cross.
        logits = self.model.joiner.step(zero_enc, pred)  # [K, width, V]
        # Renormalise over the non-blank labels: the internal LM is a distribution over emitted
        # tokens, and the blank column is an alignment decision, not a language one. Blank is the
        # last column of the logits, so dropping it leaves column i == token id i.
        logp = F.log_softmax(logits[..., : self.blank].float(), dim=-1)
        picked = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        valid = torch.arange(width, device=device).unsqueeze(0) < torch.tensor(
            lengths, device=device
        ).unsqueeze(1)
        return (picked * valid).sum(dim=1).tolist()
