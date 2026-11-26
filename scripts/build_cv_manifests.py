# Common Voice Scripted Speech (English) -> cv-train / cv-dev / cv-test / cv-dev2k manifests.
# Decodes every clip once (minutes with the process pool): MP3 headers do not give exact 16 kHz
# lengths, and the cv-train silence trim (config/trim.yaml) needs the waveform.
import argparse

from src.slices.BuildManifest.BuildCommonVoiceManifest_Command import (
    BuildCommonVoiceManifestCommand,
)
from src.slices.BuildManifest.BuildCommonVoiceManifest_Handler import build_common_voice_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Build the Common Voice manifests.")
    p.add_argument("--corpus-dir", default="data/CommonVoice", help="holds validated.tsv, clips/")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    r = build_common_voice_manifest(
        BuildCommonVoiceManifestCommand(
            corpus_dir=a.corpus_dir,
            train_out="data/manifests/cv-train.jsonl",
            dev_out="data/manifests/cv-dev.jsonl",
            test_out="data/manifests/cv-test.jsonl",
            dev_subset_out="data/manifests/cv-dev2k.jsonl",
            num_workers=a.workers,
        )
    )
    print(
        f"cv-train {r.train_rows:,} clips, {r.train_hours_untrimmed:,.1f} h -> "
        f"{r.train_hours:,.1f} h after trimming ({r.trimmed_rows:,} clips trimmed)"
    )
    print(f"cv-train dropped: {r.dropped}")
    print(f"cv-dev {r.dev_rows:,} · cv-test {r.test_rows:,} · cv-dev2k {r.dev_subset_rows:,}")


if __name__ == "__main__":
    main()
