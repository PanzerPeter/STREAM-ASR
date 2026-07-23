# src/slices/Demo/TranscriptFormat.py — presentation-only readability pass for demo output.
# The tokenizer is trained on LibriSpeech transcripts, which are upper-case and unpunctuated, so a
# raw decode reads as "MISTER QUILTER IS THE APOSTLE". That is exactly what Evaluate must score
# against the references, so the Decode slice keeps emitting it verbatim -- the cosmetic pass lives
# here, on the display path only, and never touches a scored hypothesis.
import re

# The pronoun "I", either standing alone or as the head of a contraction ("i'm", "i'll", "i've"):
# an apostrophe is a non-word character, so the trailing \b covers both. Words that merely start
# with i (in, is, it) have no boundary after the i and are left alone.
_PRONOUN_I = re.compile(r"\bi\b")


def format_transcript(text: str) -> str:
    # Upper-case corpus text -> sentence case. Deliberately conservative: casing an unpunctuated
    # stream cannot recover proper nouns or sentence boundaries, and guessing them would put words
    # in the model's mouth. Only the two transformations that are always right are applied --
    # lower-casing plus the leading capital, and the English pronoun "I".
    lowered = text.lower()
    cased = _PRONOUN_I.sub("I", lowered)
    for i, ch in enumerate(cased):
        if ch.isalpha():
            return cased[:i] + ch.upper() + cased[i + 1 :]
    return cased
