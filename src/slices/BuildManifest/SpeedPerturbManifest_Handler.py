import json
import os

from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand

# 3-way speed perturbation as a pure manifest transform: no audio is decoded here. Each source
# utterance is emitted once per speed factor; the audio itself is resampled on the fly at load time
# (LibriSpeechDataset reads the row's `speed`). num_samples is pre-corrected to the perturbed length
# so FrameBucketSampler's frame budget stays accurate -- a 0.9x row is ~11% LONGER, and bucketing
# the un-perturbed count would blow the VRAM budget it exists to cap.

_UNPERTURBED = 1.0


def _perturbed_samples(num_samples: int, speed: float) -> int:
    # sox `speed s` resamples by 1/s, so output length = input / s. Matches the resample factor
    # LibriSpeechDataset applies, keeping the sampler's frame estimate aligned with the real mel.
    return round(num_samples / speed)


def build_speed_perturb_manifest(cmd: SpeedPerturbManifestCommand) -> int:
    if not os.path.isfile(cmd.manifest_in):
        raise FileNotFoundError(cmd.manifest_in)
    os.makedirs(os.path.dirname(cmd.manifest_out) or ".", exist_ok=True)

    rows = [json.loads(line) for line in open(cmd.manifest_in, encoding="utf-8")]
    written = 0
    with open(cmd.manifest_out, "w", encoding="utf-8") as sink:
        for row in rows:
            for speed in cmd.speeds:
                out = dict(row)
                out["speed"] = speed
                if speed != _UNPERTURBED:
                    # Distinct uttid so the three copies never collide in logs or dedup, and a
                    # corrected sample count so the length-bucketed sampler sees the real duration.
                    out["uttid"] = f"{row['uttid']}_sp{speed}"
                    out["num_samples"] = _perturbed_samples(row["num_samples"], speed)
                sink.write(json.dumps(out) + "\n")
                written += 1
    return written
