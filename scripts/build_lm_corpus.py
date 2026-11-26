# One-time build of the STREAM-LM text corpus (CPU, ~1 h, network): LibriSpeech-LM + FineWeb-Edu
# (sample/10BT, ODC-BY, ~1B filtered words, shards fetched one at a time and deleted after reading)
# + cv-train sentences, with every dev/test transcript held out. Writes data/lm_text/lm-corpus.txt;
# scripts/prepare_lm_data.py packs it.
import urllib.request
from pathlib import Path

from src.slices.TrainLanguageModel.BuildLmCorpus_Command import BuildLmCorpus_Command
from src.slices.TrainLanguageModel.BuildLmCorpus_Handler import BuildLmCorpus_Handler

_FINEWEB = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
# All 14 shards of sample/10BT (~0.55B raw words per full shard, before the digit and length
# filters); the handler stops fetching once the word target is met.
_SHARDS = tuple(f"{_FINEWEB}/{i:03d}_00000.parquet" for i in range(14))
_HELDOUT = (
    "data/manifests/dev.jsonl",
    "data/manifests/dev-other.jsonl",
    "data/manifests/test.jsonl",
    "data/manifests/test-other.jsonl",
    "data/manifests/cv-dev.jsonl",
    "data/manifests/cv-test.jsonl",
)


def _download(url: str, dest: Path) -> None:
    # Written under a temporary name so an interrupted download is never read as a shard.
    part = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, part)
    part.rename(dest)


def main() -> None:
    r = BuildLmCorpus_Handler(_download).run(
        BuildLmCorpus_Command(
            librispeech_lm="data/lm_text/librispeech-lm-norm.txt.gz",
            cv_train_manifest="data/manifests/cv-train.jsonl",
            heldout_manifests=_HELDOUT,
            fineweb_urls=_SHARDS,
            fineweb_words=1_000_000_000,
            work_dir="data/lm_text/work",
            out_text="data/lm_text/lm-corpus.txt",
        )
    )
    for name in r.words:
        print(f"{name:<12} {r.lines[name]:>12,} lines {r.words[name]:>14,} words")
    print(f"held-out lines dropped: {r.heldout_dropped:,}")


if __name__ == "__main__":
    main()
