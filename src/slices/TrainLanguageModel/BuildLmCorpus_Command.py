from dataclasses import dataclass


@dataclass(frozen=True)
class BuildLmCorpus_Command:
    librispeech_lm: str  # the LibriSpeech-LM normalised corpus, plain or .gz
    cv_train_manifest: str  # its unique sentences join the corpus
    # Every dev/test manifest that is ever scored: no line equal to one of their transcripts
    # (after normalisation) enters the corpus.
    heldout_manifests: tuple[str, ...]
    fineweb_urls: tuple[str, ...]  # FineWeb-Edu parquet shards, fetched in order
    fineweb_words: int  # stop fetching once this many filtered web words are written
    work_dir: str  # per-source text and the one shard being read; emptied on success
    out_text: str  # one line per sentence, the input PrepareLmData packs

    def __post_init__(self) -> None:
        if self.fineweb_words < 0:
            raise ValueError(f"fineweb_words must be >= 0, got {self.fineweb_words}")
