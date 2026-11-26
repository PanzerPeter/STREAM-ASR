# STREAM-LM: deep-narrow causal Transformer over the LM's own tokenizer. Pre-norm blocks with
# weightless RMSNorm, grouped-query attention (QK-norm, RoPE, learnable value residual), a
# bias-free SwiGLU FFN, and untied input/output embeddings with a zero-initialised readout.
# Inference exposes a full-sequence scorer for n-best rescoring.
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.CausalGqaAttention import CausalGqaAttention, rms_norm
from src.slices.TrainLanguageModel.SwiGluFfn import SwiGluFfn


class _Block(nn.Module):
    def __init__(
        self,
        d: int,
        heads: int,
        kv_groups: int,
        ffn_expansion: int,
        dropout: float,
        value_residual: float | None,
    ) -> None:
        super().__init__()
        self.attn = CausalGqaAttention(d, heads, kv_groups, dropout, value_residual)
        self.ffn = SwiGluFfn(d, expansion=ffn_expansion, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, v = self.attn(rms_norm(x), value_residual, attn_mask)
        x = x + a
        x = x + self.ffn(rms_norm(x))
        return x, v


class StreamLmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        lm = get_config().lm
        self.vocab = lm.vocab_size
        self.eos = lm.eos_id
        self.tok_emb = nn.Embedding(self.vocab, lm.d_model)
        self.blocks = nn.ModuleList(
            _Block(
                lm.d_model,
                lm.heads,
                lm.kv_groups,
                lm.ffn_expansion,
                lm.dropout,
                value_residual=None if i == 0 else lm.value_residual_lambda,
            )
            for i in range(lm.layers)
        )
        # Untied and zero-initialised: step-0 logits are uniform (loss = ln V). Reading out through
        # the N(0, 1) embedding table instead gives logits of std ~sqrt(d_model) and a step-0 loss
        # of hundreds of nats. Untying costs vocab x d_model parameters and no compute: the readout
        # matmul is the same either way.
        self.head = nn.Linear(lm.d_model, self.vocab, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        # `segments` [B, T] gives each token its line index inside the packed window. When present,
        # attention is restricted to earlier tokens of the SAME line, so a training position sees
        # exactly what a decode-time hypothesis sees: [EOS] + its own sentence, nothing before.
        # RoPE stays absolute-in-window, which is harmless: attention only reads relative offsets.
        x = self.tok_emb(tokens)
        mask = self._document_mask(segments) if segments is not None else None
        v0 = None
        for i, blk in enumerate(self.blocks):
            x, v = blk(x, v0, mask)
            if i == 0:
                v0 = v
        return self.head(rms_norm(x))

    def _document_mask(self, segments: torch.Tensor) -> torch.Tensor:
        # [B, 1, T, T] boolean: attend to position j from position i iff j <= i and both are in the
        # same line. Broadcasts over heads.
        t = segments.shape[1]
        causal = torch.ones(t, t, dtype=torch.bool, device=segments.device).tril()
        same_line = segments.unsqueeze(2) == segments.unsqueeze(1)
        return (causal & same_line).unsqueeze(1)

    def sequence_logprob(self, ids: list[int]) -> float:
        return self.sequence_logprob_batch([list(ids)])[0]

    def sequence_logprob_batch(self, seqs: list[list[int]]) -> list[float]:
        # Score a whole n-best in ONE padded forward instead of one forward per hypothesis: the
        # rescorer calls this once per utterance, so a beam costs a single kernel launch chain and
        # a single host sync rather than beam_size of each.
        #
        # Row layout: input = [EOS] + ids (padded), target = ids + [EOS] (padded). EOS doubles as
        # BOS because LmDataset trains every line as [EOS] + line. Attention is causal, so a
        # position never sees the padding to its right; the per-row sum masks the pad positions
        # out, which makes every row exactly equal to its own single-sequence score.
        if not seqs:
            return []
        device = self.tok_emb.weight.device
        lengths = [len(s) + 1 for s in seqs]  # +1 for the EOS target
        width = max(lengths)
        inp = torch.full((len(seqs), width), self.eos, dtype=torch.long, device=device)
        tgt = torch.full((len(seqs), width), self.eos, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if s:
                ids = torch.tensor(s, dtype=torch.long, device=device)
                inp[i, 1 : len(s) + 1] = ids
                tgt[i, : len(s)] = ids
        logp = F.log_softmax(self.forward(inp).float(), dim=-1)
        picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [K, width]
        valid = torch.arange(width, device=device).unsqueeze(0) < torch.tensor(
            lengths, device=device
        ).unsqueeze(1)
        return (picked * valid).sum(dim=1).tolist()
