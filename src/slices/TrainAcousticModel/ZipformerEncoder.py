# src/slices/TrainAcousticModel/ZipformerEncoder.py
import os

import torch
import torch.nn as nn

from src.shared_kernel.MaskUtils import make_pad_mask
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.Conv2dSubsampling import Conv2dSubsampling
from src.slices.TrainAcousticModel.Resample import SimpleDownsample
from src.slices.TrainAcousticModel.ZipformerStack import ZipformerStack
from src.slices.TrainAcousticModel.BiasNorm import BiasNorm


class ZipformerEncoder(nn.Module):
    """M1 acoustic encoder. log-mel -> CMVN -> ×2 conv -> 6 multi-rate stacks -> ×2 downsample
    -> BiasNorm. Output ~25 Hz. The (features, lengths) -> (encoded, out_lengths) signature is
    the frozen contract M2 (CTC/attention) and streaming inference depend on."""

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
