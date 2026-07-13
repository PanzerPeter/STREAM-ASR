import json
from pathlib import Path

import sentencepiece as spm

from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand
from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer


def test_tokenizer_roundtrips(tmp_path: Path):
    manifest = tmp_path / "m.jsonl"
    words = "the quick brown fox jumps over a lazy dog and runs far away today".split()
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({"text": " ".join(words)}) + "\n")
    model = train_tokenizer(
        TrainTokenizerCommand(str(manifest), str(tmp_path / "sp"), vocab_size=32)
    )
    sp = spm.SentencePieceProcessor(model_file=model)
    assert sp.decode(sp.encode("the lazy dog")) == "the lazy dog"
