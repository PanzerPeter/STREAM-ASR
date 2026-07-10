# scripts/download_lm_text.py — one-time fetch of the LibriSpeech-LM normalized corpus.
import sys
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
DEST = Path("data/lm_text/librispeech-lm-norm.txt.gz")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
        return
    print(f"downloading {URL} -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done: {DEST.stat().st_size} bytes")


if __name__ == "__main__":
    sys.exit(main())
