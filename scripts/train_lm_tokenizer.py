# Train STREAM-LM's own SentencePiece unigram (lm.vocab_size, lm.eos_id) on the LM corpus (CPU,
# a few minutes). Run after build_lm_corpus.py and before prepare_lm_data.py, whose bins are ids of
# this tokenizer.
from pathlib import Path

from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainLanguageModel.TrainLmTokenizer_Command import TrainLmTokenizer_Command
from src.slices.TrainLanguageModel.TrainLmTokenizer_Handler import train_lm_tokenizer


def main() -> None:
    lm = get_config().lm
    model = train_lm_tokenizer(
        TrainLmTokenizer_Command(
            corpus_text="data/lm_text/lm-corpus.txt",
            model_prefix=str(Path(lm.tokenizer).with_suffix("")),
            vocab_size=lm.vocab_size,
            eos_id=lm.eos_id,
        )
    )
    print(f"LM tokenizer: {model}")


if __name__ == "__main__":
    main()
