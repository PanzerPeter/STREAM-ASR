# Retrain the 500-token SentencePiece unigram on every training source's transcripts. The base
# LibriSpeech manifest, not train_sp2: its speed-perturbed copies would count each line twice.
from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand
from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer


def main() -> None:
    model = train_tokenizer(
        TrainTokenizerCommand(
            manifests=("data/manifests/train.jsonl", "data/manifests/cv-train.jsonl"),
            model_prefix="data/tokenizer/bpe500",
            vocab_size=500,
        )
    )
    print(f"tokenizer: {model}")


if __name__ == "__main__":
    main()
