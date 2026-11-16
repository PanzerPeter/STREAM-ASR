# Sequence padding + chunk-visibility masks (pure fns, Shared Kernel eligible)
import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # True where the position index is >= that row's valid length, i.e. padding.
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, max_len]
    return positions >= lengths.unsqueeze(1)  # [B, max_len]


def make_chunk_mask(
    seq_len: int, chunk_size: int, left_context: int, device: torch.device
) -> torch.Tensor | None:
    """Chunk-causal visibility with a CHUNK-granular left window. True = attend.

    Query i in the chunk starting at s(i) sees key j iff s(i) - left <= j < s(i) + chunk. The window
    is anchored at the chunk start, not at i, because that is what a streaming ring buffer holds:
    a per-frame window would give each query of one chunk a different history and break
    streaming == batched. chunk_size <= 0 -> full context (None); left_context < 0 -> unlimited.
    """
    if chunk_size <= 0:
        return None
    idx = torch.arange(seq_len, device=device)
    start = (idx // chunk_size * chunk_size).unsqueeze(1)  # [T, 1] chunk start of each query
    visible = idx.unsqueeze(0) < start + chunk_size
    if left_context >= 0:
        visible &= idx.unsqueeze(0) >= start - left_context
    return visible
