# src/slices/TrainLanguageModel/PrepareLmData_Handler.py
# Text corpus -> BPE-500 token ids (+EOS per line) -> packed uint16 train/val bins. The whole
# corpus is streamed straight to disk in flushed shards, so RAM stays bounded no matter how large
# it is; uniform coverage comes from the training DataLoader shuffling the memmapped bin, not from
# an in-RAM shuffle here. The first val_words go to val.bin (a monitoring set); the remainder, up to
# subset_words, goes to train.bin (subset_words >= corpus size -> the whole corpus).
from pathlib import Path
from typing import IO, Protocol

import numpy as np

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command

# Flush the token buffer to disk every ~8M tokens (~16 MB), keeping peak RAM tiny for any corpus.
_FLUSH_TOKENS = 8_000_000


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class PrepareLmData_Handler:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tok = tokenizer
        self.eos = get_config().model.eos_id

    def run(self, cmd: PrepareLmData_Command) -> None:
        out = Path(cmd.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        val_ids: list[int] = []
        train_buf: list[int] = []
        val_words = 0
        train_words = 0
        with (
            open(cmd.source_text, "r", encoding="utf-8", errors="ignore") as src,
            open(out / "train.bin", "wb") as tf,
        ):
            for line in src:
                line = line.strip()
                if not line:
                    continue
                toks = self.tok.encode(line)
                toks.append(self.eos)
                n_words = line.count(" ") + 1
                if val_words < cmd.val_words:
                    val_ids.extend(toks)
                    val_words += n_words
                elif train_words < cmd.subset_words:
                    train_buf.extend(toks)
                    train_words += n_words
                    if len(train_buf) >= _FLUSH_TOKENS:
                        self._flush(train_buf, tf)
                else:
                    break
            self._flush(train_buf, tf)
        np.asarray(val_ids, dtype=np.uint16).tofile(out / "val.bin")

    def _flush(self, buf: list[int], fh: IO[bytes]) -> None:
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(fh)
            buf.clear()
