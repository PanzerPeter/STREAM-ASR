# src/slices/TrainAcousticModel/ZipformerBlock.py
import torch
import torch.nn as nn

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.BiasNorm import BiasNorm
from src.slices.TrainAcousticModel.SwiGluFfn import SwiGluFfn
from src.slices.TrainAcousticModel.RotaryAttention import RotaryAttention
from src.slices.TrainAcousticModel.ConvModule import ConvModule
from src.slices.TrainAcousticModel.StreamCache import AttnCache, ConvCache


class ZipformerBlock(nn.Module):
    """Macaron layer: half-FFN -> self-attention -> conv -> half-FFN, each pre-normed with
    BiasNorm and added as a residual, then a final BiasNorm."""

    def __init__(self, dim: int, num_heads: int, kernel: int | None = None) -> None:
        super().__init__()
        if kernel is None:
            kernel = get_config().model.conv_kernel_size
        self.norm_ffn1 = BiasNorm(dim)
        self.ffn1 = SwiGluFfn(dim)
        self.norm_attn = BiasNorm(dim)
        self.attn = RotaryAttention(dim, num_heads)
        self.norm_conv = BiasNorm(dim)
        self.conv = ConvModule(dim, kernel)
        self.norm_ffn2 = BiasNorm(dim)
        self.ffn2 = SwiGluFfn(dim)
        self.norm_out = BiasNorm(dim)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor, attn_visible: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))
        x = x + self.attn(self.norm_attn(x), pad_mask, attn_visible)
        x = x + self.conv(self.norm_conv(x), pad_mask)
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))
        return self.norm_out(x)

    def streaming_forward(
        self, x: torch.Tensor, attn_cache: AttnCache, conv_cache: ConvCache
    ) -> tuple[torch.Tensor, AttnCache, ConvCache]:
        # Macaron layer in streaming mode: accumulate cache in each block.
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))
        h, attn_cache = self.attn.streaming_forward(self.norm_attn(x), attn_cache)
        x = x + h
        h, conv_left = self.conv.streaming_forward(self.norm_conv(x), conv_cache.left)
        x = x + h
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))
        return self.norm_out(x), attn_cache, ConvCache(left=conv_left)
