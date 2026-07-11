# src/slices/TrainAcousticModel/AttentionDecoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.BiasNorm import BiasNorm
from src.shared_kernel.SwiGluFfn import SwiGluFfn

_MAX_TARGET_LEN = 512  # LibriSpeech-100 BPE-500 targets are well under this; positions are learned


class _SelfAttention(nn.Module):
    """Multi-head causal self-attention that exposes its value tensor, so the decoder can inject
    layer-0 values into deeper layers (value residual). Learned positions live in the embedding, so
    no RoPE here. Replaces nn.MultiheadAttention purely to make `v` reachable."""

    def __init__(
        self, dim: int, heads: int, dropout: float, value_residual_init: float = 0.0
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        # Learnable value-residual gate, init 0 (layer-0 gets no residual, so its gate is inert).
        self.res_lambda = nn.Parameter(torch.tensor(float(value_residual_init)))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        value_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, T, head_dim]
        if value_residual is not None:
            v = v + self.res_lambda * value_residual
        # True = attend: not future-masked and not a padded key. A real (non-pad) query at step i
        # always sees itself (i <= i, non-pad), so no row is fully masked -> no NaN.
        visible = (~causal_mask)[None, None] & (~tgt_pad_mask)[:, None, None, :]  # [B, 1, T, T]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=visible, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.out(out), v


class _DecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer: causal self-attn -> cross-attn to encoder memory ->
    SwiGLU FFN, each added as a residual. BiasNorm pre-norm to match the encoder."""

    def __init__(
        self, dim: int, heads: int, ffn_expansion: int, dropout: float, value_residual_init: float
    ) -> None:
        super().__init__()
        self.norm_self = BiasNorm(dim)
        self.self_attn = _SelfAttention(
            dim, heads, dropout, value_residual_init=value_residual_init
        )
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
        value_residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, v = self.self_attn(self.norm_self(x), causal_mask, tgt_pad_mask, value_residual)
        x = x + h
        h = self.norm_cross(x)
        h, _ = self.cross_attn(
            h, memory, memory, key_padding_mask=memory_pad_mask, need_weights=False
        )
        x = x + h
        return x + self.ffn(self.norm_ffn(x)), v


class _Decoder(nn.Module):
    """One directional decoder stack (shared embedding is passed in from the parent)."""

    def __init__(
        self, dim: int, num_layers: int, heads: int, ffn_expansion: int, dropout: float
    ) -> None:
        super().__init__()
        lam = get_config().model.decoder_value_residual_lambda
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(dim, heads, ffn_expansion, dropout, 0.0 if i == 0 else lam)
                for i in range(num_layers)
            ]
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
        v0: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            x, v = layer(
                x,
                causal_mask,
                tgt_pad_mask,
                memory,
                memory_pad_mask,
                value_residual=None if i == 0 else v0,
            )
            if i == 0:
                v0 = v
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
