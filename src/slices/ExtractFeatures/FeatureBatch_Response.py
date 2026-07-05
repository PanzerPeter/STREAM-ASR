# src/slices/ExtractFeatures/FeatureBatch_Response.py — output DTO (AC-009)
from dataclasses import dataclass
import torch


@dataclass
class FeatureBatch:
    features: torch.Tensor  # [B, Tmax, N_MELS] float32
    feature_lengths: torch.Tensor  # [B] long
    tokens: torch.Tensor  # [B, Umax] long
    token_lengths: torch.Tensor  # [B] long
    # Attention-decoder teacher-forcing targets (Stage B). Shapes [B, Umax+1] long.
    dec_in_l2r: torch.Tensor
    dec_out_l2r: torch.Tensor
    dec_in_r2l: torch.Tensor
    dec_out_r2l: torch.Tensor
    dec_lengths: torch.Tensor  # [B] long = token_lengths + 1
