# src/slices/BuildManifest/BuildManifest_Command.py — input DTO (AC-009)
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildManifestCommand:
    split_dir: str
    manifest_out: str
