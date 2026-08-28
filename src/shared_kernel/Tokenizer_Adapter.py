import sentencepiece as spm


class SentencePieceTokenizer:
    # sentencepiece's snake_case surface (`encode`, `decode`, the `model_file=` constructor kwarg)
    # is attached to the class at import time by a loop in its `__init__.py`, so no type checker
    # can see it. The CamelCase methods used here are the ones declared on the class and are the
    # same functions -- verified identical on bpe500 for ids, text and vocab size.
    def __init__(self, model_path: str) -> None:
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        return self._sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self._sp.DecodeIds(ids)
