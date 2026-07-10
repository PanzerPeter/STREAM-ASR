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
        self, x: torch.Tensor, value_residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a, v = self.attn(self.norm_attn(x), value_residual)
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(tokens)
        v0 = None
        for i, blk in enumerate(self.blocks):
            x, v = blk(x, value_residual=None if i == 0 else v0)
            if i == 0:
                v0 = v
        return self.head(self.norm_out(x))

    def sequence_logprob(self, ids: list[int]) -> float:
        m = get_config().model
        seq = torch.tensor([[m.sos_id] + list(ids)], device=self.tok_emb.weight.device)
        logp = F.log_softmax(self.forward(seq)[0], dim=-1)
        target = torch.tensor(list(ids) + [m.eos_id], device=logp.device)
        return float(logp[torch.arange(target.shape[0]), target].sum())

    def step_logprob(
        self, token: int, state: list[KvCache] | None
    ) -> tuple[torch.Tensor, list[KvCache]]:
        # state: (list[KvCache] per block, v0 for the current step start) or None to begin at SOS.
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
