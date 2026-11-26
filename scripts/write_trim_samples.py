# Listening check for the cv-train silence trim: writes N random trimmed clips as
# <uttid>.full.wav / <uttid>.trim.wav pairs. Listen before building the cv-train cache; a trim
# that cuts speech means raising margin_s or lowering threshold_fraction in config/trim.yaml.
import argparse
import os
import random

import soundfile as sf

from src.shared_kernel.AudioIO_Adapter import load_audio, load_manifest, load_row_audio
from src.shared_kernel.Config_Adapter import get_config


def main() -> None:
    p = argparse.ArgumentParser(description="Write before/after WAV pairs of trimmed clips.")
    p.add_argument("--out", required=True, help="an empty scratch directory")
    p.add_argument("--manifest", default="data/manifests/cv-train.jsonl")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    trimmed = [r for r in load_manifest(a.manifest) if "start_sample" in r]
    os.makedirs(a.out, exist_ok=True)
    sr = get_config().audio.sample_rate
    for row in random.Random(a.seed).sample(trimmed, min(a.n, len(trimmed))):
        full = load_audio(row["audio_filepath"])
        sf.write(os.path.join(a.out, f"{row['uttid']}.full.wav"), full.numpy(), sr)
        sf.write(os.path.join(a.out, f"{row['uttid']}.trim.wav"), load_row_audio(row).numpy(), sr)
        cut_head = row["start_sample"] / sr
        cut_tail = (full.numel() - row["start_sample"] - row["num_samples"]) / sr
        print(f"{row['uttid']}: cut {cut_head:.2f} s head, {cut_tail:.2f} s tail | {row['text']}")
    print(f"{len(trimmed):,} of the manifest's rows are trimmed; pairs written to {a.out}")


if __name__ == "__main__":
    main()
