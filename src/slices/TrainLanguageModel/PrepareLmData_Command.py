from dataclasses import dataclass


@dataclass(frozen=True)
class PrepareLmData_Command:
    source_text: str  # a plain-text corpus, one utterance per line, uppercase
    out_dir: str  # writes train.bin + val.bin here
    subset_words: int  # cap on train words; >= corpus size packs the whole corpus
    val_words: int  # first val_words become the val.bin monitoring set
