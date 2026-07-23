# src/slices/TrainLanguageModel/StreamLmModel.py
# STREAM-LM: deep-narrow causal Transformer. Reuses shared BiasNorm/SwiGluFfn + RoPE; adds GQA
# (CausalGqaAttention), QK-norm, tied embeddings, and value-residual (layer-0 values injected into
# every deeper layer). Inference exposes a full-sequence scorer (rescore) and an incremental
# next-token scorer (shallow fusion).
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.CausalGqaAttention import CausalGqaAttention, KvCache


class _Block(nn.Module):
    def __init__(
        self, d: int, heads: int, kv_groups: int, ffn_expansion: int, dropout: float, lam: float
    ) -> None:
        super().__init__()
        self.norm_attn = BiasNorm(d)
        self.attn = CausalGqaAttention(d, heads, kv_groups, dropout)
        self.attn.lam = lam
        self.norm_ffn = BiasNorm(d)
        self.ffn = SwiGluFfn(d, expansion=ffn_expansion, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, v = self.attn(self.norm_attn(x), value_residual, attn_mask)
        x = x + a
        x = x + self.ffn(self.norm_ffn(x))
        return x, v

    def step(
        self, x_t: torch.Tensor, cache: KvCache, value_residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, KvCache]:
        a, v, cache = self.attn.step(self.norm_attn(x_t), cache, value_residual)
        x_t = x_t + a
        x_t = x_t + self.ffn(self.norm_ffn(x_t))
        return x_t, v, cache


class StreamLmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        lm = get_config().lm
        self.vocab = get_config().model.decoder_vocab_size
        self.tok_emb = nn.Embedding(self.vocab, lm.d_model)
        self.blocks = nn.ModuleList(
            _Block(
                lm.d_model,
                lm.heads,
                lm.kv_groups,
                lm.ffn_expansion,
                lm.dropout,
                lam=0.0 if i == 0 else lm.value_residual_lambda,
            )
            for i in range(lm.layers)
        )
        self.norm_out = BiasNorm(lm.d_model)
        self.head = nn.Linear(lm.d_model, self.vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor | None = None) -> torch.Tensor:
        # `segments` [B, T] gives each token its line index inside the packed window. When present,
        # attention is restricted to earlier tokens of the SAME line, so a training position sees
        # exactly what a decode-time hypothesis sees: its own sentence from the start, nothing
        # before it. Without it the model spends nearly all of training conditioned on cross-line
        # context that never exists at rescoring time. RoPE stays absolute-in-window, which is
        # harmless: attention only ever reads relative offsets.
        x = self.tok_emb(tokens)
        mask = self._document_mask(segments) if segments is not None else None
        v0 = None
        for i, blk in enumerate(self.blocks):
            x, v = blk(x, None if i == 0 else v0, mask)
            if i == 0:
                v0 = v
        return self.head(self.norm_out(x))

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
        # Row layout: input = [BOS] + ids (padded), target = ids + [EOS] (padded). Attention is
        # causal, so a position never sees the padding to its right; the per-row sum masks the pad
        # positions out, which makes every row exactly equal to its own single-sequence score.
        if not seqs:
            return []
        m = get_config().model
        device = self.tok_emb.weight.device
        lengths = [len(s) + 1 for s in seqs]  # +1 for the EOS target
        width = max(lengths)
        inp = torch.full((len(seqs), width), m.eos_id, dtype=torch.long, device=device)
        tgt = torch.full((len(seqs), width), m.eos_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            inp[i, 0] = m.bos_id
            if s:
                inp[i, 1 : len(s) + 1] = torch.tensor(s, dtype=torch.long, device=device)
                tgt[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        logp = F.log_softmax(self.forward(inp), dim=-1)
        picked = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [K, width]
        valid = torch.arange(width, device=device).unsqueeze(0) < torch.tensor(
            lengths, device=device
        ).unsqueeze(1)
        return (picked * valid).sum(dim=1).tolist()

    def step_logprob(
        self, token: int, state: list[KvCache] | None
    ) -> tuple[torch.Tensor, list[KvCache]]:
        # state: (list[KvCache] per block, v0 for the current step start) or None to begin at BOS.
        device = self.tok_emb.weight.device
        lm = get_config().lm
        if state is None:
            # ModuleList.__getitem__ is typed to return the base Module; cast to recover
            # the concrete _Block so mypy sees .attn.head_dim (int), matching the codebase's
            # existing ModuleList-indexing idiom (see ZipformerStack.streaming_forward).
            head_dim = cast(_Block, self.blocks[0]).attn.head_dim
            caches = [
                KvCache.empty(1, lm.kv_groups, head_dim, device, self.tok_emb.weight.dtype)
                for _ in self.blocks
            ]
        else:
            caches = state
        x = self.tok_emb(torch.tensor([[token]], device=device))
        v0 = None
        new_caches = []
        for i, blk_module in enumerate(self.blocks):
            blk = cast(_Block, blk_module)
            x, v, c = blk.step(x, caches[i], value_residual=None if i == 0 else v0)
            if i == 0:
                v0 = v
            new_caches.append(c)
        logp = F.log_softmax(self.head(self.norm_out(x))[0, 0], dim=-1)
        return logp, new_caches
