# src/slices/ComputeCmvn/ComputeCmvn_Command.py — input DTO (AC-009)
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeCmvnCommand:
    manifest: str
    cmvn_out: str
    max_utts: int | None = None
    sample_frac: float | None = (
        None  # random fraction of rows; global CMVN converges well before 960h
    )
    seed: int = 0
