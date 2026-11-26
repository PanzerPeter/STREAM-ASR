import json
import math

import numpy as np
import soundfile as sf

from src.slices.BuildManifest.BuildCommonVoiceManifest_Command import (
    BuildCommonVoiceManifestCommand,
)
from src.slices.BuildManifest.BuildCommonVoiceManifest_Handler import build_common_voice_manifest

_SR = 48000  # Common Voice's rate: the manifest counts samples after resampling to 16 kHz
_HEADER = ["client_id", "path", "sentence_id", "sentence", "up_votes", "down_votes"]


def _silence_tone_silence() -> np.ndarray:
    quiet = 1e-4 * np.random.default_rng(0).standard_normal(2 * _SR)
    tone = 0.3 * np.sin(2 * math.pi * 220 * np.arange(int(1.5 * _SR)) / _SR)
    return np.concatenate([quiet, tone, quiet]).astype(np.float32)


def _tsv(path, rows) -> None:
    lines = ["\t".join(_HEADER)] + ["\t".join([c, p, "id", s, "2", "0"]) for c, p, s in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_filters_leakage_trims_train_and_keeps_heldout_whole(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    noise = (0.2 * np.random.default_rng(1).standard_normal(3 * _SR)).astype(np.float32)
    for name in ("d1", "t1", "spk", "sent", "digit", "keep"):
        sf.write(clips / f"{name}.wav", _silence_tone_silence(), _SR)
    sf.write(clips / "noisy.wav", noise, _SR)
    sf.write(clips / "fast.wav", noise[: _SR // 2], _SR)
    (clips / "bad.wav").write_bytes(b"not audio at all")

    _tsv(tmp_path / "dev.tsv", [("dev_speaker", "d1.wav", "Held-out sentence.")])
    _tsv(tmp_path / "test.tsv", [("test_speaker", "t1.wav", "Another one!")])
    _tsv(
        tmp_path / "validated.tsv",
        [
            ("dev_speaker", "d1.wav", "Held-out sentence."),  # the dev clip itself
            ("dev_speaker", "spk.wav", "A fresh sentence"),  # a dev voice
            ("alice", "sent.wav", "held out sentence"),  # a dev sentence, other spelling
            ("alice", "digit.wav", "Room 101"),
            ("alice", "keep.wav", "Keep me, please."),
            ("bob", "noisy.wav", "Noisy clip here"),
            ("bob", "bad.wav", "Broken file here"),
            ("bob", "fast.wav", "far too many words for half a second of audio to hold"),
        ],
    )
    out = {k: str(tmp_path / f"cv-{k}.jsonl") for k in ("train", "dev", "test", "dev2k")}
    r = build_common_voice_manifest(
        BuildCommonVoiceManifestCommand(
            str(tmp_path), out["train"], out["dev"], out["test"], out["dev2k"], 1, num_workers=2
        )
    )

    assert r.dropped == {
        "heldout_clip": 1,
        "heldout_speaker": 1,
        "unspellable": 1,
        "heldout_sentence": 1,
        "undecodable": 1,
        "too_fast": 1,
    }
    train = {row["uttid"]: row for row in map(json.loads, open(out["train"], encoding="utf-8"))}
    assert list(train) == ["keep", "noisy"]  # sorted by uttid
    keep, noisy = train["keep"], train["noisy"]
    assert keep["text"] == "KEEP ME PLEASE"
    # 5.5 s at 16 kHz, trimmed to the tone plus 0.5 s each side (~2.5 s), tone untouched.
    assert (
        0 < keep["start_sample"] <= 2 * 16000
        and keep["start_sample"] + keep["num_samples"] >= 3.5 * 16000
    )
    assert 2.4 * 16000 <= keep["num_samples"] <= 2.6 * 16000
    assert "start_sample" not in noisy and noisy["num_samples"] == 3 * 16000
    assert r.trimmed_rows == 1 and r.train_hours < r.train_hours_untrimmed

    dev = [json.loads(line) for line in open(out["dev"], encoding="utf-8")]
    assert dev == [
        {
            "uttid": "d1",
            "audio_filepath": str(clips / "d1.wav"),
            "text": "HELD OUT SENTENCE",
            "num_samples": int(5.5 * 16000),
        }
    ]
    assert open(out["dev2k"], encoding="utf-8").read() == open(out["dev"], encoding="utf-8").read()
    assert r.test_rows == 1
