# src/slices/BuildManifest/BuildManifest_Handler.py
import json
import os
from glob import glob

import soundfile as sf

from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand

# torchaudio 2.11 dropped its metadata backend along with decoding; soundfile.info reads
# the FLAC header for frame counts without pulling in TorchCodec/FFmpeg.


def build_manifest(cmd: BuildManifestCommand) -> int:
    if not os.path.isdir(cmd.split_dir):
        raise FileNotFoundError(cmd.split_dir)

    os.makedirs(os.path.dirname(cmd.manifest_out) or ".", exist_ok=True)

    rows = 0
    with open(cmd.manifest_out, "w", encoding="utf-8") as sink:
        for trans_path in sorted(glob(f"{cmd.split_dir}/**/*.trans.txt", recursive=True)):
            chapter_dir = os.path.dirname(trans_path)
            for line in open(trans_path, encoding="utf-8"):
                uttid, text = line.strip().split(" ", 1)
                audio_path = os.path.join(chapter_dir, f"{uttid}.flac")
                num_samples = sf.info(audio_path).frames
                sink.write(
                    json.dumps(
                        {
                            "uttid": uttid,
                            "audio_filepath": audio_path,
                            "text": text,
                            "num_samples": num_samples,
                        }
                    )
                    + "\n"
                )
                rows += 1
    return rows
