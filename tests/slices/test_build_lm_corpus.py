import gzip
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.slices.TrainLanguageModel.BuildLmCorpus_Command import BuildLmCorpus_Command
from src.slices.TrainLanguageModel.BuildLmCorpus_Handler import BuildLmCorpus_Handler


def _manifest(path: Path, texts: list[str]) -> str:
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return str(path)


def test_corpus_filters_holds_out_and_interleaves(tmp_path):
    ls = tmp_path / "ls.txt.gz"
    with gzip.open(ls, "wt", encoding="utf-8") as f:
        f.write("".join(f"BOOK LINE NUMBER {w}\n" for w in ("ONE", "TWO", "THREE", "FOUR")))
        f.write("THE HELD OUT LINE\n")
    shard = tmp_path / "src.parquet"
    docs = [
        "Modern words like smartphone appear. Short. In 2024 nothing counts.",
        "Another web sentence here! The held-out line? Final web words remain",
    ]
    pq.write_table(pa.table({"text": docs, "id": ["a", "b"]}), shard)
    fetched: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        shutil.copy(shard, dest)

    work, out = tmp_path / "work", tmp_path / "corpus.txt"
    r = BuildLmCorpus_Handler(fetch).run(
        BuildLmCorpus_Command(
            librispeech_lm=str(ls),
            cv_train_manifest=_manifest(tmp_path / "cv.jsonl", ["CV SAYS HI", "CV SAYS HI"]),
            heldout_manifests=(_manifest(tmp_path / "dev.jsonl", ["THE HELD OUT LINE"]),),
            fineweb_urls=("u0", "u1", "u2"),
            fineweb_words=15,
            work_dir=str(work),
            out_text=str(out),
        )
    )

    lines = out.read_text().splitlines()
    assert "THE HELD OUT LINE" not in lines and r.heldout_dropped == 2  # one book, one web
    web = [line for line in lines if not line.startswith(("BOOK", "CV"))]
    # The digit sentence and the one-word fragment are dropped. Shard one yields 13 words, so the
    # 15-word target is met by the first sentence of the second, and the third is never fetched.
    assert web == [
        "MODERN WORDS LIKE SMARTPHONE APPEAR",
        "ANOTHER WEB SENTENCE HERE",
        "FINAL WEB WORDS REMAIN",
        "MODERN WORDS LIKE SMARTPHONE APPEAR",
    ]
    assert fetched == ["u0", "u1"] and r.words["fineweb"] == 18
    assert lines.count("CV SAYS HI") == 1  # unique sentences only
    assert r.lines == {"librispeech": 4, "fineweb": 4, "cv": 1}
    assert sorted(lines) == sorted(
        [f"BOOK LINE NUMBER {w}" for w in ("ONE", "TWO", "THREE", "FOUR")] + web + ["CV SAYS HI"]
    )
    assert list(work.iterdir()) == []  # per-source files and the shard are gone


def test_interleave_keeps_every_prefix_proportional(tmp_path):
    from src.slices.TrainLanguageModel.BuildLmCorpus_Handler import _interleave, _SourceFile

    big, small = _SourceFile(tmp_path / "big.txt", set()), _SourceFile(
        tmp_path / "small.txt", set()
    )
    for i in range(64 * 20):
        big.add(f"BIG {i}")
    for i in range(64 * 2):
        small.add(f"SMALL {i}")
    big.close()
    small.close()
    out = tmp_path / "out.txt"
    _interleave([big, small], out)
    lines = out.read_text().splitlines()
    assert len(lines) == 64 * 22
    half = lines[: len(lines) // 2]
    assert sum(line.startswith("SMALL") for line in half) == 64  # half the small source, in order
    assert [line for line in lines if line.startswith("BIG")] == [f"BIG {i}" for i in range(1280)]
