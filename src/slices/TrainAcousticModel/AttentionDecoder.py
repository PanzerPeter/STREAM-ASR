# src/slices/TrainAcousticModel/AttentionDecoder.py
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn

_MAX_TARGET_LEN = 512  # LibriSpeech-100 BPE-500 targets are well under this; positions are learned


class _DecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer: causal self-attn -> cross-attn to encoder memory ->
    SwiGLU FFN, each added as a residual. BiasNorm pre-norm to match the encoder."""

    def __init__(self, dim: int, heads: int, ffn_expansion: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = BiasNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = BiasNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = BiasNorm(dim)
        self.ffn = SwiGluFfn(dim, expansion=ffn_expansion, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm_self(x)
        h, _ = self.self_attn(
            h, h, h, attn_mask=causal_mask, key_padding_mask=tgt_pad_mask, need_weights=False
        )
        x = x + h
        h = self.norm_cross(x)
        h, _ = self.cross_attn(
            h, memory, memory, key_padding_mask=memory_pad_mask, need_weights=False
        )
        x = x + h
        return x + self.ffn(self.norm_ffn(x))


class _Decoder(nn.Module):
    """One directional decoder stack (shared embedding is passed in from the parent)."""

    def __init__(
        self, dim: int, num_layers: int, heads: int, ffn_expansion: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(dim, heads, ffn_expansion, dropout) for _ in range(num_layers)]
        )
        self.norm_out = BiasNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal_mask, tgt_pad_mask, memory, memory_pad_mask)
        return self.norm_out(x)


class BiTransformerDecoder(nn.Module):
    """U2++ bidirectional decoder. A shared token+position embedding and output projection feed a
    left (L2R) decoder and a smaller right (R2L, reversed-target) decoder. Cross-attends the
    256-dim encoder memory, projected to decoder_dim. Trained by teacher forcing; used to rescore
    in Phase 2."""

    def __init__(self) -> None:
        super().__init__()
        m = get_config().model
        dim = m.decoder_dim
        self.embed = nn.Embedding(m.decoder_vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, _MAX_TARGET_LEN, dim))
        self.dropout = nn.Dropout(m.decoder_dropout)
        self.memory_proj = (
            nn.Linear(m.encoder_dims[-1], dim) if m.encoder_dims[-1] != dim else nn.Identity()
        )
        self.left = _Decoder(
            dim, m.decoder_left_layers, m.decoder_heads, m.decoder_ffn_expansion, m.decoder_dropout
        )
        self.right = _Decoder(
            dim, m.decoder_right_layers, m.decoder_heads, m.decoder_ffn_expansion, m.decoder_dropout
        )
        self.out_proj = nn.Linear(dim, m.decoder_vocab_size)

    def forward(
        self,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        ys_in: torch.Tensor,
        ys_pad_mask: torch.Tensor,
        reverse: bool,
    ) -> torch.Tensor:
        u = ys_in.shape[1]
        if u > _MAX_TARGET_LEN:
            raise ValueError(f"target length {u} exceeds _MAX_TARGET_LEN {_MAX_TARGET_LEN}")
        x = self.dropout(self.embed(ys_in) + self.pos[:, :u])
        mem = self.memory_proj(memory)
        # Bool causal mask (True = masked) matches the bool key_padding_mask dtype, so
        # MultiheadAttention does not warn about mismatched mask types; step i still cannot
        # attend to any step > i (True strictly above the diagonal).
        causal = torch.ones(u, u, dtype=torch.bool, device=ys_in.device).triu(1)
        decoder = self.right if reverse else self.left
        h = decoder(x, causal, ys_pad_mask, mem, memory_pad_mask)
        return self.out_proj(h)
