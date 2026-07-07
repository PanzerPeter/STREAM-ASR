# src/slices/TrainAcousticModel/ZipformerStack.py
from typing import cast

import torch
import torch.nn as nn

from src.shared_kernel.MaskUtils import make_chunk_mask, make_pad_mask
from src.slices.TrainAcousticModel.Resample import SimpleDownsample, SimpleUpsample
from src.slices.TrainAcousticModel.StreamCache import AttnCache, ConvCache
from src.slices.TrainAcousticModel.ZipformerBlock import ZipformerBlock


class ZipformerStack(nn.Module):
    """One Zipformer stack. Projects to the stack width, optionally downsamples to a lower
    frame rate, runs the blocks there, upsamples back to the base rate, and mixes with the
    stack input through a learnable scalar bypass. Output frame count == input frame count."""

    def __init__(
        self, dim_in: int, dim: int, num_layers: int, downsample: int, num_heads: int
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim_in, dim) if dim_in != dim else nn.Identity()
        self.downsample = SimpleDownsample(downsample) if downsample > 1 else None
        self.upsample = SimpleUpsample(downsample) if downsample > 1 else None
        self.blocks = nn.ModuleList([ZipformerBlock(dim, num_heads) for _ in range(num_layers)])
        self.bypass = nn.Parameter(torch.tensor(0.5))  # residual↔processed interpolation

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        base_pad_mask: torch.Tensor,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        x = self.in_proj(x)
        residual = x
        base_len = x.shape[1]

        if self.downsample is not None:
            x, ds_lengths = self.downsample(x, lengths)
            pad_mask = make_pad_mask(ds_lengths, x.shape[1])
            # Chunk size is expressed in base-rate frames; scale to this stack's downsampled rate.
            local_chunk = max(1, chunk_size // self.downsample.factor) if chunk_size > 0 else 0
        else:
            pad_mask = base_pad_mask
            local_chunk = chunk_size

        attn_visible = (
            make_chunk_mask(x.shape[1], local_chunk, x.device) if chunk_size > 0 else None
        )
        for block in self.blocks:
            x = block(x, pad_mask, attn_visible)

        if self.upsample is not None:
            x = self.upsample(x, out_len=base_len)

        bypass = self.bypass.clamp(0.0, 1.0)
        return residual + bypass * (x - residual)

    def streaming_forward(
        self,
        x: torch.Tensor,
        attn_caches: list[AttnCache],
        conv_caches: list[ConvCache],
    ) -> tuple[torch.Tensor, list[AttnCache], list[ConvCache]]:
        # Chunks are aligned to downsample factor, so downsample never straddles boundary.
        # Bypass and up/down-sample are per-frame linear ops. Streaming reduces to running
        # each block (causal conv + KV-cached attn) at the stack's rate.
        x = self.in_proj(x)
        residual = x
        base_len: int = x.shape[1]
        if self.downsample is not None:
            frame_len: int = x.shape[1]
            lengths: torch.Tensor = torch.tensor([frame_len], device=x.device)
            x, _ = self.downsample(x, lengths)
        new_ac: list[AttnCache] = []
        new_cc: list[ConvCache] = []
        for i, (ac, cc) in enumerate(zip(attn_caches, conv_caches)):
            block = cast(ZipformerBlock, self.blocks[i])
            x, ac_new, cc_new = block.streaming_forward(x, ac, cc)
            new_ac.append(ac_new)
            new_cc.append(cc_new)
        if self.upsample is not None:
            x = self.upsample(x, out_len=base_len)
        bypass = self.bypass.clamp(0.0, 1.0)
        return residual + bypass * (x - residual), new_ac, new_cc
