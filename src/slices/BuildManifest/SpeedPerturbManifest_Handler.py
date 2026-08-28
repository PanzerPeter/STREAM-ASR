import hashlib
import json
import os

from src.slices.BuildManifest.SpeedPerturbManifest_Command import SpeedPerturbManifestCommand

# Paired 2x speed perturbation as a pure manifest transform: no audio is decoded here. Each source
# utterance is emitted twice -- once untouched, once at ONE factor drawn from cmd.speeds -- and the
# audio itself is resampled at feature-extraction time (the row carries its `speed`). num_samples is
# pre-corrected to the perturbed length so FrameBucketSampler's frame budget stays accurate: a 0.9x
# row is ~11% LONGER, and bucketing the un-perturbed count would blow the VRAM budget it exists to
# cap.
#
# 2x rather than icefall's 3x cross product because the binding constraint here is disk, not epochs:
# the fp16 mel cache is 53 GB per copy of the 961 h corpus. Drawing one factor per utterance keeps
# both directions represented across the corpus at half the storage.

_UNPERTURBED = 1.0


def _perturbed_samples(num_samples: int, speed: float) -> int:
    # sox `speed s` resamples by 1/s, so output length = input / s. Matches the resample factor
    # applied at extraction, keeping the sampler's frame estimate aligned with the real mel.
    return round(num_samples / speed)


def _factor_for(uttid: str, speeds: tuple[float, ...], seed: int) -> float:
    # hashlib, never the builtin hash(): Python randomises string hashing per process, so hash()
    # would give a different manifest on every rebuild -- and the feature cache is indexed by row
    # order, so a rebuild that reshuffled the factors would silently pair every utterance with
    # someone else's mel.
    digest = hashlib.sha1(f"{uttid}:{seed}".encode()).hexdigest()
    return speeds[int(digest, 16) % len(speeds)]


def build_speed_perturb_manifest(cmd: SpeedPerturbManifestCommand) -> int:
    if not os.path.isfile(cmd.manifest_in):
        raise FileNotFoundError(cmd.manifest_in)
    os.makedirs(os.path.dirname(cmd.manifest_out) or ".", exist_ok=True)

    rows = [json.loads(line) for line in open(cmd.manifest_in, encoding="utf-8")]
    written = 0
    with open(cmd.manifest_out, "w", encoding="utf-8") as sink:
        for row in rows:
            clean = dict(row)
            clean["speed"] = _UNPERTURBED
            sink.write(json.dumps(clean) + "\n")

            speed = _factor_for(row["uttid"], cmd.speeds, cmd.seed)
            out = dict(row)
            out["speed"] = speed
            # Distinct uttid so the two copies never collide in logs or dedup, and a corrected
            # sample count so the length-bucketed sampler sees the real duration.
            out["uttid"] = f"{row['uttid']}_sp{speed}"
            out["num_samples"] = _perturbed_samples(row["num_samples"], speed)
            sink.write(json.dumps(out) + "\n")
            written += 2
    return written
