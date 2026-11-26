# Raw sentence -> the acoustic model's output alphabet: upper-case A-Z, apostrophe, single spaces.
# One function serves the Common Voice manifests and the LM corpus, so acoustic transcripts and LM
# text are written the same way and the tokenizer sees one alphabet.
import re
import unicodedata

# A symbol with a spoken form that is not its spelling ("3", "&", "%") cannot be transcribed without
# number or symbol verbalisation, which is out of scope. Dropping the symbol would leave a label
# missing words the speaker said, so the whole line is rejected instead.
_UNSPELLED = re.compile(r"[0-9&%$£€@#+=]")
_APOSTROPHES = re.compile(r"[‘’‛ʼ`´]")
# ASCII hyphen, the Unicode hyphen/dash block (U+2010..U+2015) and the slash all join or separate
# words in written text, so they become word boundaries rather than vanishing ("well-known").
_WORD_JOINERS = re.compile(r"[-‐-―/]")
_OUTSIDE_ALPHABET = re.compile(r"[^A-Z' ]")


def normalise_transcript(text: str) -> str | None:
    """The line in the output alphabet, or None when it cannot be transcribed faithfully."""
    if _UNSPELLED.search(text):
        return None
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    spaced = _WORD_JOINERS.sub(" ", _APOSTROPHES.sub("'", stripped))
    letters = _OUTSIDE_ALPHABET.sub("", spaced.upper())
    # An apostrophe at a word edge is a quotation mark; inside a word it is a contraction.
    words = [w.strip("'") for w in letters.split()]
    line = " ".join(w for w in words if w)
    return line or None
