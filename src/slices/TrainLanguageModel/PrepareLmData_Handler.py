# src/slices/TrainLanguageModel/PrepareLmData_Handler.py
# Text corpus -> BPE-500 token ids (+EOS per line) -> packed uint16 train/val bins. The whole
# corpus is streamed straight to disk in flushed shards, so RAM stays bounded no matter how large
# it is; uniform coverage comes from the training DataLoader shuffling the memmapped bin, not from
# an in-RAM shuffle here. The first val_words go to val.bin (a monitoring set); the remainder, up to
# subset_words, goes to train.bin (subset_words >= corpus size -> the whole corpus).
#
# Tokenizing the full LibriSpeech-LM corpus (~40M lines / 4 GB) is a multi-minute job; without
# progress on the terminal it looks frozen, so it gets killed mid-write (partial train.bin, no
# val.bin). Loguru progress lines every flush make the run observable, matching the training slice.
import time
from pathlib import Path
from typing import IO, Protocol

import numpy as np

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
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
        log = configure_logging()
        out = Path(cmd.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        val_ids: list[int] = []
        train_buf: list[int] = []
        val_words = 0
        train_words = 0
        train_tokens = 0  # total flushed to train.bin (train_buf is cleared on each flush)
        lines = 0
        val_written = False
        start = time.perf_counter()
        log.info(
            f"Preparing LM data from {cmd.source_text} -> {out} "
            f"(val {cmd.val_words:,} words, train cap {cmd.subset_words:,} words)"
        )
        with (
            open(cmd.source_text, "r", encoding="utf-8", errors="ignore") as src,
            open(out / "train.bin", "wb") as tf,
        ):
            for line in src:
                line = line.strip()
                if not line:
                    continue
                lines += 1
                toks = self.tok.encode(line)
                toks.append(self.eos)
                n_words = line.count(" ") + 1
                if val_words < cmd.val_words:
                    val_ids.extend(toks)
                    val_words += n_words
                elif train_words < cmd.subset_words:
                    # Val set is complete the first time we spill into the train branch: write it
                    # now so an interrupt during the (much longer) train pass can't lose val.bin.
                    if not val_written:
                        self._flush_val(val_ids, out)
                        log.info(f"Wrote val.bin: {len(val_ids):,} tokens")
                        val_written = True
                    train_buf.extend(toks)
                    train_words += n_words
                    if len(train_buf) >= _FLUSH_TOKENS:
                        train_tokens += len(train_buf)
                        self._flush(train_buf, tf)
                        elapsed = time.perf_counter() - start
                        rate = lines / max(elapsed, 1e-9)
                        log.info(
                            f"lines {lines:>12,} │ train {train_words:>13,} words / "
                            f"{train_tokens:>13,} tokens ({train_tokens * 2 / 1e6:8.1f} MB) │ "
                            f"{rate:8.0f} lines/s"
                        )
                else:
                    break
            train_tokens += len(train_buf)
            self._flush(train_buf, tf)
        if not val_written:  # corpus smaller than val_words -> val loop never crossed into train
            self._flush_val(val_ids, out)
        log.info(
            f"Done in {time.perf_counter() - start:.0f}s: {lines:,} lines │ "
            f"train {train_tokens:,} tokens -> {out / 'train.bin'} │ "
            f"val {len(val_ids):,} tokens -> {out / 'val.bin'}"
        )

    def _flush_val(self, val_ids: list[int], out: Path) -> None:
        np.asarray(val_ids, dtype=np.uint16).tofile(out / "val.bin")

    def _flush(self, buf: list[int], fh: IO[bytes]) -> None:
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(fh)
            buf.clear()
