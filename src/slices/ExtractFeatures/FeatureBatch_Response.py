# src/slices/ExtractFeatures/FeatureBatch_Response.py — output DTO (AC-009)
from dataclasses import dataclass
import torch


@dataclass
class FeatureBatch:
    features: torch.Tensor  # [B, Tmax, N_MELS] float32
    feature_lengths: torch.Tensor  # [B] long
    tokens: torch.Tensor  # [B, Umax] long
    token_lengths: torch.Tensor  # [B] long
