from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand
from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.shared_kernel.Config_Adapter import get_config


def test_tokenizer_trains_and_roundtrips(tmp_path):
    vocab_size = get_config().model.vocab_size
    prefix = str(tmp_path / "bpe500")
    model = train_tokenizer(
        TrainTokenizerCommand(
            manifest="data/manifests/train.jsonl", model_prefix=prefix, vocab_size=vocab_size
        )
    )

    tok = SentencePieceTokenizer(model)
    assert tok.vocab_size == vocab_size

    text = "THE QUICK BROWN FOX DIDN'T JUMP"
    assert tok.decode(tok.encode(text)) == text
