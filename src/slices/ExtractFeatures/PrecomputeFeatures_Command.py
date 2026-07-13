# src/slices/ExtractFeatures/PrecomputeFeatures_Command.py — input DTO (AC-009)
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecomputeFeaturesCommand:
    manifest: str
    split: str
    cache_dir: str
    num_workers: int = 8
