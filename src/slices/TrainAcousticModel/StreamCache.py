# Explicit, caller-carried streaming state. Keeping it out of the modules makes the encoder a pure
# function of (chunk, cache), which is what makes the equivalence gate tractable.
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ConvCache:
    left: torch.Tensor  # [B, kernel-1, dim] causal left context for the depthwise conv


@dataclass
class AttnCache:
    k: torch.Tensor  # [B, heads, L, head_dim] (already RoPE-embedded at absolute positions)
    v: torch.Tensor
    seen: int  # frames consumed at this block's rate -> RoPE pos_offset


@dataclass
class FrontendCache:
    in_tail: torch.Tensor  # [B, 2, n_mels] last inputs for causal conv1
    mid_tail: torch.Tensor  # [B, channels, 2, freq_mid] last post-conv1 frames for causal conv2


@dataclass
class StreamCache:
    frontend: FrontendCache
    attn: list[AttnCache] = field(default_factory=list)
    conv: list[ConvCache] = field(default_factory=list)

    @staticmethod
    def init(encoder, batch_size: int, device: torch.device | None = None) -> "StreamCache":
        dev = device or encoder.cmvn_mean.device
        n_mels = encoder.cmvn_mean.shape[0]
        front = encoder.frontend
        attn: list[AttnCache] = []
        conv: list[ConvCache] = []
        for stack in encoder.stacks:
            for block in stack.blocks:
                heads, hd = block.attn.num_heads, block.attn.head_dim
                dim = heads * hd
                attn.append(
                    AttnCache(
                        k=torch.zeros(batch_size, heads, 0, hd, device=dev),
                        v=torch.zeros(batch_size, heads, 0, hd, device=dev),
                        seen=0,
                    )
                )
                conv.append(
                    ConvCache(left=torch.zeros(batch_size, block.conv.kernel - 1, dim, device=dev))
                )
        return StreamCache(
            frontend=FrontendCache(
                in_tail=torch.zeros(batch_size, front.time_pad, n_mels, device=dev),
                mid_tail=torch.zeros(
                    batch_size, front.conv1.out_channels, front.time_pad, front.freq_mid, device=dev
                ),
            ),
            attn=attn,
            conv=conv,
        )
