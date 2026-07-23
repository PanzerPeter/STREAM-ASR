from src.slices.Demo.TranscriptFormat import format_transcript


def test_upper_case_corpus_text_becomes_sentence_case():
    # The tokenizer emits LibriSpeech-style upper case; the demo must show readable text.
    out = format_transcript("MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES")
    assert out == "Mister quilter is the apostle of the middle classes"


def test_pronoun_i_stays_capital_but_other_i_words_do_not():
    out = format_transcript("HE SAID I'M SURE I WILL IF IT IS IN THE INDEX")
    assert out == "He said I'm sure I will if it is in the index"


def test_apostrophes_and_partial_prefixes_survive():
    # Live partials are mid-utterance prefixes: capitalise the first letter, change nothing else.
    assert format_transcript("QUILTER'S MANN") == "Quilter's mann"


def test_empty_and_symbol_only_text_is_returned_unchanged():
    assert format_transcript("") == ""
    assert format_transcript("'") == "'"
