# src/shared_kernel/Tokenizer_Adapter.py
import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)
