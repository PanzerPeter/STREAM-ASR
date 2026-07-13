# scripts/train_tokenizer.py — retrain BPE-500 on the 960h train transcripts
from src.slices.BuildManifest.TrainTokenizer_Command import TrainTokenizerCommand
from src.slices.BuildManifest.TrainTokenizer_Handler import train_tokenizer


def main() -> None:
    model = train_tokenizer(
        TrainTokenizerCommand(
            manifest="data/manifests/train.jsonl",
            model_prefix="data/tokenizer/bpe500",
            vocab_size=500,
        )
    )
    print(f"tokenizer: {model}")


if __name__ == "__main__":
    main()
