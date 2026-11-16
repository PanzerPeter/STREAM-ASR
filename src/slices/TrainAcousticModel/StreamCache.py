# Explicit, caller-carried streaming state. Keeping it out of the modules makes the encoder a pure
# function of (chunk, cache), which is what makes the equivalence gate tractable.
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FrontendCache:
    in_tail: torch.Tensor  # [B, 1, n_mels] last normalised mel frame, conv1's left context
    mid_tail: torch.Tensor  # [B, C, 1, freq_mid] last conv1 output frame, dw2's left context


@dataclass
class BlockCache:
    conv: torch.Tensor  # [B, kernel-1, hf] last frames of u, the causal conv's left context
    # The last `left` frames of PRE-RoPE keys and their values, plus which of them are real.
    # Pre-RoPE because positions are re-assigned window-locally every chunk, so the rotation never
    # sees an unbounded index.
    k: torch.Tensor  # [B, H, L, D]
    v: torch.Tensor
    key_valid: torch.Tensor  # [B, L] bool


@dataclass
class StreamCache:
    frontend: FrontendCache
    blocks: list[BlockCache]
    left_context: int  # 25 Hz frames of attention history kept; -1 = unlimited
