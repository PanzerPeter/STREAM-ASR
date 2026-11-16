import os
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.MaskUtils import make_chunk_mask, make_pad_mask
from src.slices.TrainAcousticModel.DwSepSubsampling import FRONTEND_STRIDE, DwSepSubsampling
from src.slices.TrainAcousticModel.FusedBranchBlock import (
    FusedBranchBlock,
    attention_mask,
    mlp_width,
    rms_norm,
)
from src.slices.TrainAcousticModel.StreamCache import BlockCache, FrontendCache, StreamCache

# Frame-rate divisor of each stack relative to 25 Hz: S0 25, S1 12.5, S2 6.25, S3 12.5.
_RATE_DIV = (1, 2, 4, 2)
# Chunks and left contexts count 25 Hz frames and must split evenly at the slowest rate (6.25 Hz).
CHUNK_LCM = 4
# The input is padded to the total stride, so every pooling step averages frames the network
# computed from real or mean-filled input, never an activation-space zero.
_TOTAL_STRIDE = FRONTEND_STRIDE * CHUNK_LCM


class _Stack(nn.Module):
    """proj (Identity when the width does not change) -> rms_norm(x + skip) -> blocks. The entry
    norm is weightless, so whatever proj's scale, spectrum or rank, the stack starts at RMS 1."""

    def __init__(self, d_in: int, dim: int, blocks: int) -> None:
        super().__init__()
        self.proj: nn.Module = nn.Identity() if d_in == dim else nn.Linear(d_in, dim, bias=False)
        self.blocks = nn.ModuleList([FusedBranchBlock(dim) for _ in range(blocks)])

    def block_list(self) -> list[FusedBranchBlock]:
        return cast(list[FusedBranchBlock], list(self.blocks))


class S2Encoder(nn.Module):
    """Raw log-mel -> CMVN -> DwSepSubsampling (25 Hz) -> S0 -> pool -> S1 -> pool -> S2 ->
    upsample + skip(S1) -> S3 -> rms_norm = memory at 12.5 Hz (MODEL_ARCHITECTURE.md §4.1). With
    S2 at 0 blocks its stage is the linear 6.25 Hz path proj -> rms_norm -> proj.

    Every path takes RAW log-mel and fills every position past a row's length with the CMVN mean,
    so batch padding, the pad to the total stride and the streaming tail are all the data mean.
    """

    cmvn_mean: torch.Tensor
    cmvn_std: torch.Tensor

    def __init__(self, cmvn_path: str | None = "data/features/cmvn.pt") -> None:
        super().__init__()
        m = get_config().model
        dims = list(m.stack_dims)
        self.frontend = DwSepSubsampling(dims[0])
        d_in = [dims[0], dims[0], dims[1], dims[2]]
        self.stacks = nn.ModuleList([_Stack(d_in[i], dims[i], m.stack_blocks[i]) for i in range(4)])
        self.output_dim = dims[3]
        # 25 Hz frames per output frame: the memory is S3's, at 12.5 Hz.
        self.out_div = _RATE_DIV[3]
        self.last_exit_rms: list[torch.Tensor] = []
        self.last_skip_ratio: torch.Tensor | None = None
        mean, std = self._load_cmvn(cmvn_path)
        self.register_buffer("cmvn_mean", mean)  # [n_mels]
        self.register_buffer("cmvn_std", std)  # [n_mels]

    @staticmethod
    def _load_cmvn(cmvn_path: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        if cmvn_path is not None and os.path.isfile(cmvn_path):
            stats = torch.load(cmvn_path, map_location="cpu")
            return stats["mean"], stats["std"]
        # Identity normalisation when stats are absent (tests); a checkpoint carries the real ones.
        n_mels = get_config().audio.n_mels
        return torch.zeros(n_mels), torch.ones(n_mels)

    def stack_list(self) -> list[_Stack]:
        return cast(list[_Stack], list(self.stacks))

    @property
    def mel_frames_per_output(self) -> int:
        return FRONTEND_STRIDE * self.out_div

    def output_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        return -(-mel_lengths // self.mel_frames_per_output)

    @staticmethod
    def check_window(chunk_size: int, left_context: int) -> None:
        if chunk_size % CHUNK_LCM or (left_context >= 0 and left_context % CHUNK_LCM):
            raise ValueError(
                f"chunk_size {chunk_size} and left_context {left_context} must be multiples of "
                f"{CHUNK_LCM} (25 Hz frames); left_context -1 = unlimited"
            )

    def _normalise(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # 0 after CMVN is the raw per-bin mean: the only constant ever written into log-mel.
        x = (features - self.cmvn_mean) / self.cmvn_std
        return x.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _record(self, exits: list[torch.Tensor], skip_ratio: torch.Tensor) -> None:
        # Detached scalars for train/stack_rms/*; the trainer reads them at log_every only.
        with torch.no_grad():
            self.last_exit_rms = [e.detach().float().pow(2).mean().sqrt() for e in exits]
            self.last_skip_ratio = skip_ratio.detach().float()

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        chunk_size: int = 0,
        left_context: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, 80] raw log-mel, [B] mel lengths -> (memory [B, T_out, d], out_lengths [B])."""
        self.check_window(chunk_size, left_context)
        pad = -features.shape[1] % _TOTAL_STRIDE
        features = F.pad(features, (0, 0, 0, pad))
        valid = ~make_pad_mask(lengths, features.shape[1])
        x = self.frontend(self._normalise(features, valid))
        len25 = -(-lengths // FRONTEND_STRIDE)

        def run(stack: _Stack, x: torch.Tensor, rate: int) -> torch.Tensor:
            n = -(-len25 // rate)
            key_valid = ~make_pad_mask(n, x.shape[1])
            window = make_chunk_mask(
                x.shape[1], chunk_size // rate, _per_rate(left_context, rate), x.device
            )
            mask = attention_mask(key_valid, key_valid, window)
            for block in stack.block_list():
                x = block(x, mask)
            return x

        s = self.stack_list()
        s0 = run(s[0], _enter(s[0].proj(x)), 1)
        s1 = run(s[1], _enter(s[1].proj(_pool(s0))), 2)
        s2 = run(s[2], _enter(s[2].proj(_pool(s1))), 4)
        deep = _upsample(s[3].proj(s2), s1.shape[1])
        s3 = run(s[3], _enter(deep + s1), 2)
        self._record([s0, s1, s2, s3], deep.pow(2).mean().sqrt() / s1.pow(2).mean().sqrt())
        return rms_norm(s3), self.output_lengths(lengths)

    def init_stream(
        self, batch_size: int, left_context: int, device: torch.device | None = None
    ) -> StreamCache:
        self.check_window(0, left_context)
        dev = device or self.cmvn_mean.device
        n_mels = self.cmvn_mean.shape[0]
        f = self.frontend
        blocks: list[BlockCache] = []
        for stack in self.stack_list():
            for block in stack.block_list():
                conv = torch.zeros(batch_size, block.kernel - 1, mlp_width(block.dim), device=dev)
                kv = torch.zeros(batch_size, block.heads, 0, block.head_dim, device=dev)
                blocks.append(
                    BlockCache(
                        conv=conv,
                        k=kv,
                        v=kv.clone(),
                        key_valid=torch.zeros(batch_size, 0, dtype=torch.bool, device=dev),
                    )
                )
        return StreamCache(
            frontend=FrontendCache(
                in_tail=torch.zeros(batch_size, 1, n_mels, device=dev),
                mid_tail=torch.zeros(batch_size, f.conv1.out_channels, 1, f.freq_mid, device=dev),
            ),
            blocks=blocks,
            left_context=left_context,
        )

    def streaming_forward(
        self, features_chunk: torch.Tensor, cache: StreamCache, n_valid: int | None = None
    ) -> tuple[torch.Tensor, StreamCache]:
        """One chunk of raw log-mel, a multiple of 4*CHUNK_LCM frames. Frames from `n_valid` on are
        mean-filled and masked as keys, exactly as batched forward treats a row past its length.
        Returns ALL output frames of the chunk; the caller keeps the ones it wants."""
        t = features_chunk.shape[1]
        if t % _TOTAL_STRIDE:
            raise ValueError(f"streaming chunk must be a multiple of {_TOTAL_STRIDE} mel frames")
        n_valid = t if n_valid is None else n_valid
        b = features_chunk.shape[0]
        valid = (torch.arange(t, device=features_chunk.device) < n_valid).expand(b, t)
        x, in_tail, mid_tail = self.frontend.streaming_forward(
            self._normalise(features_chunk, valid), cache.frontend.in_tail, cache.frontend.mid_tail
        )
        cache.frontend = FrontendCache(in_tail=in_tail, mid_tail=mid_tail)
        valid25 = valid[:, ::FRONTEND_STRIDE]
        i = 0

        def run(stack: _Stack, x: torch.Tensor, rate: int) -> torch.Tensor:
            nonlocal i
            key_valid = valid25[:, ::rate]
            left = _per_rate(cache.left_context, rate)
            for block in stack.block_list():
                x, cache.blocks[i] = block.streaming_forward(x, cache.blocks[i], key_valid, left)
                i += 1
            return x

        s = self.stack_list()
        s0 = run(s[0], _enter(s[0].proj(x)), 1)
        s1 = run(s[1], _enter(s[1].proj(_pool(s0))), 2)
        s2 = run(s[2], _enter(s[2].proj(_pool(s1))), 4)
        out = run(s[3], _enter(_upsample(s[3].proj(s2), s1.shape[1]) + s1), 2)
        return rms_norm(out), cache


def _enter(x: torch.Tensor) -> torch.Tensor:
    # The residual stream is fp32 from every stack entry on: under autocast proj emits bf16, and a
    # block that sees two input dtypes compiles two graphs (and burns dynamo's recompile budget).
    return rms_norm(x.float())


def _per_rate(frames25: int, rate: int) -> int:
    return frames25 if frames25 < 0 else frames25 // rate


def _pool(x: torch.Tensor) -> torch.Tensor:
    b, t, c = x.shape
    return x.view(b, t // 2, 2, c).mean(dim=2)


def _upsample(x: torch.Tensor, length: int) -> torch.Tensor:
    return x.repeat_interleave(2, dim=1)[:, :length]
