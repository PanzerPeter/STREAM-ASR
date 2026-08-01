from dataclasses import dataclass


@dataclass(frozen=True)
class BuildManifestCommand:
    split_dir: str
    manifest_out: str
