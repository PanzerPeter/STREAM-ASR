# src/slices/TrainAcousticModel/ZipformerEncoder.py
import math
import os
from typing import cast

import torch
import torch.nn as nn

from src.shared_kernel.MaskUtils import make_pad_mask
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.Conv2dSubsampling import Conv2dSubsampling
from src.slices.TrainAcousticModel.Resample import SimpleDownsample
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack
from src.shared_kernel.BiasNorm import BiasNorm
from src.slices.TrainAcousticModel.StreamCache import FrontendCache, StreamCache


class ZipformerEncoder(nn.Module):
    """Acoustic encoder. log-mel -> CMVN -> ×2 conv -> 6 multi-rate stacks -> ×2 downsample
    -> BiasNorm. Output ~25 Hz. The (features, lengths) -> (encoded, out_lengths) signature is
    the frozen contract the CTC/attention heads and streaming inference depend on."""

    cmvn_mean: torch.Tensor
    cmvn_std: torch.Tensor

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        model = get_config().model
        self.frontend = Conv2dSubsampling()

        stacks = []
        dim_in = model.encoder_dims[0]
        for dim, layers, factor, heads in zip(
            model.encoder_dims,
            model.encoder_layers,
            model.encoder_downsampling,
            model.encoder_heads,
        ):
            stacks.append(ZipformerStack(dim_in, dim, layers, factor, heads))
            dim_in = dim
        self.stacks = nn.ModuleList(stacks)

        self.final_downsample = SimpleDownsample(model.final_downsample)
        self.out_norm = BiasNorm(model.encoder_dims[-1])

        mean, std = self._load_cmvn(cmvn_path)
        self.register_buffer("cmvn_mean", mean)  # [n_mels]
        self.register_buffer("cmvn_std", std)  # [n_mels]

    @staticmethod
    def _load_cmvn(cmvn_path: str | None):
        if cmvn_path is not None and os.path.isfile(cmvn_path):
            stats = torch.load(cmvn_path, map_location="cpu")
            return stats["mean"], stats["std"]
        # Identity normalization when stats are absent (tests / first run).
        n_mels = get_config().audio.n_mels
        return torch.zeros(n_mels), torch.ones(n_mels)

    @property
    def output_dim(self) -> int:
        return get_config().model.encoder_dims[-1]

    def forward(self, features: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 0):
        x = (features - self.cmvn_mean) / self.cmvn_std

        x, lengths = self.frontend(x, lengths)  # ×2, base rate
        pad_mask = make_pad_mask(lengths, x.shape[1])
        for stack in self.stacks:
            x = stack(x, lengths, pad_mask, chunk_size)  # base rate + length preserved

        x, out_lengths = self.final_downsample(x, lengths)  # ×2 -> ~25 Hz
        x = self.out_norm(x)
        return x, out_lengths

    def chunk_lcm(self) -> int:
        # Base-rate chunk boundaries must land cleanly at every stack's rate and the final
        # downsample. Downsampling factors are lcm'd to ensure streaming input boundaries
        # align with frame boundaries at all subsampled rates.
        model = get_config().model
        factors = list(model.encoder_downsampling) + [model.final_downsample]
        lcm = 1
        for f in factors:
            lcm = lcm * f // math.gcd(lcm, f)
        return lcm

    def streaming_forward(
        self, features_chunk: torch.Tensor, cache: StreamCache
    ) -> tuple[torch.Tensor, StreamCache]:
        # Process a feature-rate chunk through the encoder (frontend + stacks + final_downsample)
        # with causal context carried in cache. Returns (memory_chunk, updated_cache).
        # Callers feed chunks that are multiples of 2*chunk_lcm() so every stack's downsample
        # aligns; see test_streaming_forward_equivalence for the exactness guarantee.
        x = (features_chunk - self.cmvn_mean) / self.cmvn_std
        x, in_tail, mid_tail = self.frontend.streaming_forward(
            x, cache.frontend.in_tail, cache.frontend.mid_tail
        )
        cache.frontend = FrontendCache(in_tail=in_tail, mid_tail=mid_tail)
        i = 0
        for stack in self.stacks:
            stack = cast(ZipformerStack, stack)
            n = len(stack.blocks)
            x, new_ac, new_cc = stack.streaming_forward(
                x, cache.attn[i : i + n], cache.conv[i : i + n]
            )
            cache.attn[i : i + n] = new_ac
            cache.conv[i : i + n] = new_cc
            i += n
        lengths = torch.tensor([x.shape[1]], device=x.device)
        x, _ = self.final_downsample(x, lengths)
        x = self.out_norm(x)
        return x, cache
