# LibriSpeech-LM + FineWeb-Edu + Common Voice sentences -> one plain-text LM corpus, every line in
# the acoustic model's output alphabet (Transcript_Transform), so the LM scores hypotheses written
# exactly as the decoder writes them. LibriSpeech-LM alone is book English up to ~1920; the web
# slice carries modern vocabulary, and the Common Voice sentences match the cv test domain.
#
# The sources are written to their own files first, then interleaved in small blocks in proportion
# to their sizes, so any prefix of the corpus is a proportional sample of all three: val.bin (the
# first val_words) monitors every source, and a subset_words cap trims all three evenly.
import gzip
import re
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from pathlib import Path
from typing import IO, Any

import pyarrow.parquet as pq

from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Transcript_Transform import normalise_transcript
from src.slices.TrainLanguageModel.BuildLmCorpus_Command import BuildLmCorpus_Command
from src.slices.TrainLanguageModel.BuildLmCorpus_Response import BuildLmCorpus_Response

# Web text splits into sentences at terminal punctuation and line breaks. Fragments shorter than
# this are headings, list items and navigation, not sentences anyone would say.
_SENTENCE_END = re.compile(r"[.?!\n]+")
_MIN_WEB_WORDS = 3
# Small enough that the first val_words of the corpus touch every source (Common Voice is ~1 % of
# the lines), large enough that the merge loop is not per line.
_BLOCK_LINES = 64
_PARQUET_BATCH_ROWS = 1024
_SHARD_NAME = "fineweb-shard.parquet"
_SOURCE_NAMES = ("librispeech", "fineweb", "cv")

# (url, destination) -> the shard on disk at destination.
ShardFetch = Callable[[str, Path], None]


class _SourceFile:
    def __init__(self, path: Path, heldout: set[str]) -> None:
        self.path = path
        self.lines = self.words = self.heldout_dropped = 0
        self._heldout = heldout
        self._sink: IO[str] = open(path, "w", encoding="utf-8")

    def add(self, line: str) -> None:
        if line in self._heldout:
            self.heldout_dropped += 1
            return
        self._sink.write(line + "\n")
        self.lines += 1
        self.words += line.count(" ") + 1

    def close(self) -> None:
        self._sink.close()


def _open_text(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def _sentences(document: str) -> Iterator[str]:
    for piece in _SENTENCE_END.split(document):
        line = normalise_transcript(piece)
        if line is not None and line.count(" ") + 1 >= _MIN_WEB_WORDS:
            yield line


def _interleave(sources: list[_SourceFile], out: Path) -> None:
    # Always advance the source that is furthest behind its own share, one block at a time.
    live = [s for s in sources if s.lines > 0]
    readers = {s.path: open(s.path, encoding="utf-8") for s in live}
    emitted = {s.path: 0 for s in live}
    with open(out, "w", encoding="utf-8") as sink:
        while live:
            src = min(live, key=lambda s: emitted[s.path] / s.lines)
            take = min(_BLOCK_LINES, src.lines - emitted[src.path])
            for _ in range(take):
                sink.write(readers[src.path].readline())
            emitted[src.path] += take
            live = [s for s in live if emitted[s.path] < s.lines]
    for reader in readers.values():
        reader.close()


class BuildLmCorpus_Handler:
    def __init__(self, fetch: ShardFetch) -> None:
        self._fetch = fetch

    def run(self, cmd: BuildLmCorpus_Command) -> BuildLmCorpus_Response:
        log = configure_logging()
        for path in (cmd.librispeech_lm, cmd.cv_train_manifest, *cmd.heldout_manifests):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        work = Path(cmd.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        heldout = {
            line
            for m in cmd.heldout_manifests
            for row in load_manifest(m)
            if (line := normalise_transcript(row["text"])) is not None
        }
        log.info(f"LM corpus: {len(heldout):,} held-out dev/test sentences excluded")

        sources = {name: _SourceFile(work / f"{name}.txt", heldout) for name in _SOURCE_NAMES}
        self._librispeech(cmd.librispeech_lm, sources["librispeech"])
        log.info(f"librispeech: {sources['librispeech'].words:,} words")
        self._fineweb(cmd, work, sources["fineweb"], log)
        cv_lines = {row["text"] for row in load_manifest(cmd.cv_train_manifest)}
        for line in sorted(cv_lines):
            sources["cv"].add(line)
        for src in sources.values():
            src.close()

        out = Path(cmd.out_text)
        out.parent.mkdir(parents=True, exist_ok=True)
        _interleave(list(sources.values()), out)
        for src in sources.values():
            src.path.unlink()
        response = BuildLmCorpus_Response(
            lines={k: s.lines for k, s in sources.items()},
            words={k: s.words for k, s in sources.items()},
            heldout_dropped=sum(s.heldout_dropped for s in sources.values()),
        )
        log.info(f"LM corpus -> {out}: {response}")
        return response

    def _librispeech(self, path: str, sink: _SourceFile) -> None:
        with _open_text(path) as src:
            for raw in src:
                line = normalise_transcript(raw)
                if line is not None:
                    sink.add(line)

    def _fineweb(self, cmd: BuildLmCorpus_Command, work: Path, sink: _SourceFile, log: Any) -> None:
        if cmd.fineweb_words == 0:
            return
        with closing(self._web_lines(cmd.fineweb_urls, work, log)) as lines:
            for line in lines:
                sink.add(line)
                if sink.words >= cmd.fineweb_words:
                    log.info(f"fineweb: {sink.words:,} words, target reached")
                    return
        log.warning(
            f"fineweb: shards exhausted at {sink.words:,} of {cmd.fineweb_words:,} words; "
            "add shard URLs to reach the target"
        )

    def _web_lines(self, urls: tuple[str, ...], work: Path, log: Any) -> Generator[str, None, None]:
        # One shard on disk at a time (2.15 GB each), deleted as soon as it is read, also when the
        # word target stops the read part-way (closing() runs the finally).
        shard = work / _SHARD_NAME
        for url in urls:
            log.info(f"fineweb: fetching {url}")
            self._fetch(url, shard)
            try:
                for batch in pq.ParquetFile(shard).iter_batches(
                    batch_size=_PARQUET_BATCH_ROWS, columns=["text"]
                ):
                    for document in batch.column(0).to_pylist():
                        yield from _sentences(document)
            finally:
                shard.unlink(missing_ok=True)
