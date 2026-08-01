from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedPerturbManifestCommand:
    manifest_in: str
    manifest_out: str
    # sox-style speed factors. 1.0 = untouched original; <1 slows (longer), >1 speeds (shorter).
    # 0.9/1.0/1.1 is the icefall 3-way policy that lifts LibriSpeech WER at no extra epoch cost.
    speeds: tuple[float, ...] = (0.9, 1.0, 1.1)
