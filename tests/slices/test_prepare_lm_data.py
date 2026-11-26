import numpy as np
import pytest

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command
from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler


class _WordTok:
    # One id per word, never the EOS id; the vocab size is whatever the test declares.
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [2 + sum(map(ord, w)) % 1000 for w in text.split()]


def _contains_subseq(haystack: list[int], needle: list[int]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def test_prepare_writes_uint16_bins(tmp_path):
    lm = get_config().lm
    src = tmp_path / "corpus.txt"
    src.write_text("\n".join(["HELLO WORLD THIS IS A TEST"] * 200) + "\n")
    cmd = PrepareLmData_Command(
        source_text=str(src),
        out_dir=str(tmp_path / "lm_data"),
        subset_words=600,
        val_words=100,
    )
    PrepareLmData_Handler(_WordTok(lm.vocab_size)).run(cmd)
    train = np.memmap(tmp_path / "lm_data" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(tmp_path / "lm_data" / "val.bin", dtype=np.uint16, mode="r")
    assert train.shape[0] > 0 and val.shape[0] > 0
    assert lm.eos_id in set(train.tolist())  # line separators written
    assert train.max() < lm.vocab_size


def test_prepare_packs_whole_corpus_not_just_head(tmp_path):
    # Each line is distinct; with a cap far above the corpus size the LAST line's tokens must land
    # in train.bin. A head-only prefix read would miss the tail -> this locks whole-corpus coverage.
    lines = [f"UNIQUE LINE NUMBER {i} ALPHA BRAVO CHARLIE" for i in range(400)]
    src = tmp_path / "corpus.txt"
    src.write_text("\n".join(lines) + "\n")
    tok = _WordTok(get_config().lm.vocab_size)
    cmd = PrepareLmData_Command(
        source_text=str(src),
        out_dir=str(tmp_path / "lm_data"),
        subset_words=10**9,  # exceeds the toy corpus -> whole corpus
        val_words=20,
    )
    PrepareLmData_Handler(tok).run(cmd)
    train = np.memmap(tmp_path / "lm_data" / "train.bin", dtype=np.uint16, mode="r").tolist()
    assert _contains_subseq(train, tok.encode(lines[-1]))


def test_a_tokenizer_of_another_size_is_rejected():
    # Bins written with a tokenizer the model's table does not match would train on ids the
    # decode-time scorer never produces.
    with pytest.raises(ValueError):
        PrepareLmData_Handler(_WordTok(500))
