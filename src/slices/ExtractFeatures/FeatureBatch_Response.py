from dataclasses import dataclass
import torch


@dataclass
class FeatureBatch:
    features: torch.Tensor  # [B, Tmax, N_MELS] float32
    feature_lengths: torch.Tensor  # [B] long
    tokens: torch.Tensor  # [B, Umax] long
    token_lengths: torch.Tensor  # [B] long

    def pin_memory(self) -> "FeatureBatch":
        # The DataLoader's pin_memory worker only walks tensors, mappings and (named)tuples; a
        # plain dataclass matches none of those branches and would come back untouched, leaving
        # pin_memory=True paying for the pinning thread while handing the trainer pageable tensors
        # -- on which .to(device, non_blocking=True) silently degrades to a blocking copy.
        # hasattr(data, "pin_memory") is torch's documented opt-in, checked before the containers.
        return FeatureBatch(
            self.features.pin_memory(),
            self.feature_lengths.pin_memory(),
            self.tokens.pin_memory(),
            self.token_lengths.pin_memory(),
        )
