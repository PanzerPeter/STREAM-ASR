# STREAM-LM's own SentencePiece unigram, trained on the LM corpus. Rescoring reads hypothesis text,
# so the LM's vocab is independent of the acoustic BPE-500 and can be sized for language modelling:
# fewer tokens per word means more text per training step and shorter rows to score.
import os

import sentencepiece as spm

from src.slices.TrainLanguageModel.TrainLmTokenizer_Command import TrainLmTokenizer_Command


def train_lm_tokenizer(cmd: TrainLmTokenizer_Command) -> str:
    if not os.path.isfile(cmd.corpus_text):
        raise FileNotFoundError(cmd.corpus_text)
    os.makedirs(os.path.dirname(cmd.model_prefix) or ".", exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=cmd.corpus_text,
        model_prefix=cmd.model_prefix,
        vocab_size=cmd.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        input_sentence_size=cmd.sample_lines,
        shuffle_input_sentence=True,
        # unk is SentencePiece's mandatory id 0; EOS is a control symbol encode() never emits, so
        # it marks line ends only where PrepareLmData and the scorer place it. No BOS (EOS is
        # the BOS) and no pad (rows are masked, never padded with a learned id).
        unk_id=0,
        eos_id=cmd.eos_id,
        bos_id=-1,
        pad_id=-1,
        num_threads=os.cpu_count() or 1,
    )
    return f"{cmd.model_prefix}.model"
