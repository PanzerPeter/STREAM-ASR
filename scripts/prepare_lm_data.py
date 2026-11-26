# Pack the LM corpus into LM-tokenizer id bins (data/lm_data/{train,val}.bin). Re-run after every
# LM tokenizer change: the bins are token ids.
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainLanguageModel.PrepareLmData_Command import PrepareLmData_Command
from src.slices.TrainLanguageModel.PrepareLmData_Handler import PrepareLmData_Handler


def main() -> None:
    lm = get_config().lm
    PrepareLmData_Handler(SentencePieceTokenizer(lm.tokenizer)).run(
        PrepareLmData_Command(
            "data/lm_text/lm-corpus.txt", "data/lm_data", lm.subset_words, lm.val_words
        )
    )


if __name__ == "__main__":
    main()
