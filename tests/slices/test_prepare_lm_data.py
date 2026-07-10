import numpy as np

from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command
from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.shared_kernel.Config_Adapter import get_config


def _contains_subseq(haystack: list[int], needle: list[int]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def test_prepare_writes_uint16_bins(tmp_path):
    src = tmp_path / "corpus.txt"
    src.write_text("\n".join(["HELLO WORLD THIS IS A TEST"] * 200) + "\n")
    tok = SentencePieceTokenizer("data/tokenizer/bpe500.model")
    cmd = PrepareLmData_Command(
        source_text=str(src),
        out_dir=str(tmp_path / "lm_data"),
        subset_words=600,
        val_words=100,
    )
    PrepareLmData_Handler(tok).run(cmd)
    train = np.memmap(tmp_path / "lm_data" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(tmp_path / "lm_data" / "val.bin", dtype=np.uint16, mode="r")
    assert train.shape[0] > 0 and val.shape[0] > 0
    eos = get_config().model.eos_id
    assert eos in set(train.tolist())  # utterance separators written
    assert train.max() < get_config().model.decoder_vocab_size


def test_prepare_packs_whole_corpus_not_just_head(tmp_path):
    # Each line is distinct; with a cap far above the corpus size the LAST line's tokens must land
    # in train.bin. A head-only prefix read would miss the tail -> this locks whole-corpus coverage.
    lines = [f"UNIQUE LINE NUMBER {i} ALPHA BRAVO CHARLIE" for i in range(400)]
    src = tmp_path / "corpus.txt"
    src.write_text("\n".join(lines) + "\n")
    tok = SentencePieceTokenizer("data/tokenizer/bpe500.model")
    cmd = PrepareLmData_Command(
        source_text=str(src),
        out_dir=str(tmp_path / "lm_data"),
        subset_words=10**9,  # exceeds the toy corpus -> whole corpus
        val_words=20,
    )
    PrepareLmData_Handler(tok).run(cmd)
    train = np.memmap(tmp_path / "lm_data" / "train.bin", dtype=np.uint16, mode="r").tolist()
    tail = tok.encode(lines[-1])
    assert _contains_subseq(train, tail)
