import pytest

from src.shared_kernel.Transcript_Transform import normalise_transcript


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("She'll be all right.", "SHE'LL BE ALL RIGHT"),
        ("Café “déjà vu”, naïve", "CAFE DEJA VU NAIVE"),
        ("It’s a well-known — and/or old — fact", "IT'S A WELL KNOWN AND OR OLD FACT"),
        ("'Quoted,' he said; 'twas", "QUOTED HE SAID TWAS"),
        ("Straße", "STRASSE"),
        ("  spaced   out  ", "SPACED OUT"),
        ("THE PIPER'S SON", "THE PIPER'S SON"),  # LibriSpeech text is a fixed point
    ],
)
def test_normalises_to_the_output_alphabet(raw, expected):
    assert normalise_transcript(raw) == expected


@pytest.mark.parametrize("raw", ["Born in 1984", "Smith & Wesson", "50% off", "...", "—", ""])
def test_rejects_unspelled_symbols_and_empty_lines(raw):
    assert normalise_transcript(raw) is None
