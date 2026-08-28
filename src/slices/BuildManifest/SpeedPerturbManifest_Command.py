from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedPerturbManifestCommand:
    manifest_in: str
    manifest_out: str
    # sox-style speed factors to draw ONE perturbed copy from, per utterance. <1 slows (longer
    # audio), >1 speeds (shorter). Every utterance also keeps its unperturbed original, so the
    # output is exactly 2x the input regardless of how many factors are listed here -- that is the
    # disk budget: 961 h of fp16 mel is 53 GB, and a third copy is another 53 GB.
    speeds: tuple[float, ...] = (0.9, 1.1)
    # Factor assignment is hash(uttid, seed), so the manifest is byte-identical across rebuilds.
    # The feature cache is indexed by row order, so a reshuffle here silently mispairs every mel.
    seed: int = 1234
