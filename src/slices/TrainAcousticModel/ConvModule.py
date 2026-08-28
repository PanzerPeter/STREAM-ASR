import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.BiasNorm import BiasNorm


class ConvModule(nn.Module):
    """Conformer/Zipformer depthwise conv module: pointwise GLU expand -> causal depthwise conv
    -> per-frame BiasNorm -> SiLU -> pointwise project. The conv reads no future frames (left pad
    only) and the norm is per-frame, so a streaming chunk is bit-for-bit the full-sequence result.
    The per-frame norm is load-bearing: a GroupNorm over time would tie the result to the whole
    sequence and break that equivalence."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel, padding=0, groups=dim)
        self.norm = BiasNorm(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()

    @staticmethod
    def _pointwise(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # A kernel-1 Conv1d IS a Linear over the channel axis, but it insists on [B, C, T] while
        # every activation the block carries is [B, T, C]. Running it as a convolution therefore
        # costs a transpose each way -- and a transposed tensor is not contiguous, so cuDNN copies
        # it -- on top of an implicit-GEMM path that is slower than cuBLAS for the 1x1 case.
        # Measured: 51 convolution_backward calls per training step (13.9 ms) and 42 wmma conv
        # kernels (5.9 ms) for what is three convolutions per block, two of them kernel 1.
        #
        # Calling F.linear through a squeezed view of the same parameter runs the cuBLAS kernel on
        # the layout the block already has, while the module -- and therefore every checkpoint key
        # and shape, including the BEST-RQ warm start -- stays exactly as it was. Same trick as
        # TransducerJoiner._readout: keep the parameter canonical, reshape at the call site.
        return F.linear(x, conv.weight.squeeze(-1), conv.bias)

    def _depthwise_causal(self, x: torch.Tensor) -> torch.Tensor:
        # The one genuine convolution here, and the only op that needs [B, C, T].
        padded = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))  # left context only, no future
        return self.depthwise(padded).transpose(1, 2)  # [B, T, C]

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = F.glu(self._pointwise(self.pointwise1, x), dim=-1)  # [B, T, C]
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)  # keep padding out of the conv window
        x = self._depthwise_causal(x)
        x = self.activation(self.norm(x))  # BiasNorm is per frame over channels
        return self._pointwise(self.pointwise2, x)  # [B, T, C]

    def streaming_forward(
        self, x: torch.Tensor, cache_left: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cache_left holds the previous chunk's trailing kernel-1 glu-output frames, replacing the
        # zero left-pad the padded forward uses at sequence start. Per-frame norm makes this exact.
        x = F.glu(self._pointwise(self.pointwise1, x), dim=-1)  # [B, T, C]
        padded = torch.cat([cache_left, x], dim=1)
        new_left = padded[:, -(self.kernel - 1) :].detach()
        y = self.depthwise(padded.transpose(1, 2)).transpose(1, 2)  # valid conv over cache+chunk
        y = self.activation(self.norm(y))  # [B, T, C]
        return self._pointwise(self.pointwise2, y), new_left
