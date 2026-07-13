# BEST-RQ target generator (Chiu et al. 2022): FROZEN random linear projection + FROZEN L2-normed
# random codebook turn input features into discrete labels the encoder learns to predict.
# Nothing trains here — targets are a fixed, seeded input function; simpler and more stable
# than a learned quantizer (SP4).
import torch
import torch.nn as nn


class RandomProjectionQuantizer(nn.Module):
    proj: torch.Tensor
    codebook: torch.Tensor

    def __init__(self, in_dim: int, codebook_size: int, codebook_dim: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        proj = torch.empty(in_dim, codebook_dim)
        nn.init.xavier_uniform_(proj, generator=g)
        codebook = torch.randn(codebook_size, codebook_dim, generator=g)
        codebook = codebook / (codebook.norm(dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("proj", proj)
        self.register_buffer("codebook", codebook)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x @ self.proj  # [B, T, codebook_dim]
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        # Both p and codebook are unit-norm, so max cosine == min L2 == nearest codebook entry.
        sim = p @ self.codebook.t()  # [B, T, codebook_size]
        return sim.argmax(dim=-1)
