# src/shared_kernel/MaskUtils.py — sequence padding masks (pure fn, Shared Kernel eligible)
import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # True where the position index is >= that row's valid length, i.e. padding.
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, max_len]
    return positions >= lengths.unsqueeze(1)  # [B, max_len]


def make_chunk_mask(seq_len: int, chunk_size: int, device: torch.device) -> torch.Tensor:
    # Chunk-causal visibility: query i sees keys in its own chunk and all earlier chunks.
    # This is the U2++ dynamic-chunk regularizer — the same weights then run offline
    # (full) or streaming (small chunk). True = attend. chunk_size <= 0 -> full context.
    if chunk_size <= 0:
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    idx = torch.arange(seq_len, device=device)
    chunk_of = idx // chunk_size  # [seq_len]
    return chunk_of.unsqueeze(1) >= chunk_of.unsqueeze(0)  # [i, j] True iff chunk(j) <= chunk(i)
