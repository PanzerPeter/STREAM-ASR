# src/slices/BuildManifest/TrainTokenizer_Handler.py
import json
import os
import tempfile

import sentencepiece as spm

from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand


def train_tokenizer(cmd: TrainTokenizerCommand) -> str:
    if not os.path.isfile(cmd.manifest):
        raise FileNotFoundError(cmd.manifest)

    os.makedirs(os.path.dirname(cmd.model_prefix) or ".", exist_ok=True)

    # SentencePiece trains from a raw text file; extract transcripts once.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as text_file:
        for line in open(cmd.manifest, encoding="utf-8"):
            text_file.write(json.loads(line)["text"] + "\n")
        corpus_path = text_file.name

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=cmd.model_prefix,
        vocab_size=cmd.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
    )
    os.unlink(corpus_path)
    return f"{cmd.model_prefix}.model"
