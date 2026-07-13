# src/slices/BuildManifest/BuildManifest_Handler.py
import json
import multiprocessing as mp
import os
from glob import glob

import soundfile as sf

from src.slices.BuildManifest.BuildManifest_Command import BuildManifestCommand

# torchaudio 2.11 dropped its metadata backend; soundfile.info reads the FLAC header for frame
# counts without pulling in TorchCodec/FFmpeg. Header reads are IO-bound, so a process pool over
# ~281k utterances turns minutes of serial probing into seconds.


def _probe(task: tuple[str, str, str]) -> tuple[str, str, str, int]:
    uttid, audio_path, text = task
    return uttid, audio_path, text, sf.info(audio_path).frames


def build_manifest(cmd: BuildManifestCommand) -> int:
    if not os.path.isdir(cmd.split_dir):
        raise FileNotFoundError(cmd.split_dir)
    os.makedirs(os.path.dirname(cmd.manifest_out) or ".", exist_ok=True)

    tasks: list[tuple[str, str, str]] = []
    for trans_path in glob(f"{cmd.split_dir}/**/*.trans.txt", recursive=True):
        chapter_dir = os.path.dirname(trans_path)
        for line in open(trans_path, encoding="utf-8"):
            uttid, text = line.strip().split(" ", 1)
            tasks.append((uttid, os.path.join(chapter_dir, f"{uttid}.flac"), text))

    # fork (the platform default) would inherit torch's threads when preloaded by an earlier test
    # module and warns/risks deadlock; spawn is clean and matches PrecomputeFeatures_Handler.
    with mp.get_context("spawn").Pool() as pool:
        results = pool.map(_probe, tasks, chunksize=256)
    results.sort(key=lambda r: r[0])  # deterministic order by uttid

    with open(cmd.manifest_out, "w", encoding="utf-8") as sink:
        for uttid, audio_path, text, num_samples in results:
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
    return len(results)
