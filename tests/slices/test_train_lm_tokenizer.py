import sentencepiece as spm

from src.slices.TrainLanguageModel.TrainLmTokenizer_Command import TrainLmTokenizer_Command
from src.slices.TrainLanguageModel.TrainLmTokenizer_Handler import train_lm_tokenizer


def test_lm_tokenizer_reserves_eos_and_never_emits_it(tmp_path):
    words = "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG WHILE SEVEN TIRED WIZARDS WATCH".split()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(" ".join(words[i % 7 : i % 7 + 6]) for i in range(300)) + "\n")
    model = train_lm_tokenizer(
        TrainLmTokenizer_Command(
            corpus_text=str(corpus),
            model_prefix=str(tmp_path / "lm"),
            vocab_size=30,
            eos_id=1,
        )
    )
    sp = spm.SentencePieceProcessor(model_file=model)
    assert sp.GetPieceSize() == 30
    assert sp.eos_id() == 1 and sp.bos_id() == -1
    # EOS is a control symbol: only PrepareLmData and the scorer place it.
    assert 1 not in sp.EncodeAsIds("THE QUICK BROWN FOX")
