# Common Voice Scripted Speech -> cv-train / cv-dev / cv-test JSONL manifests in the LibriSpeech
# manifest format. cv-train is `validated.tsv` with every dev/test clip, speaker and sentence taken
# out, so no Common Voice score is ever measured on audio, a voice or a sentence the model trained
# on. Common Voice sentences are recorded by many speakers each, so the sentence filter is the
# large one (about a third of validated).
#
# Every clip is decoded once here: MP3 headers do not give exact 16 kHz sample counts, and the
# silence trim needs the waveform anyway. The trim is stored in the row (`start_sample`,
# `num_samples`) so the feature cache, the uncached loader and the sampler all see one span.
import csv
import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from typing import Any

import torch

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Logging_Adapter import configure_logging
from src.shared_kernel.Transcript_Transform import normalise_transcript
from src.slices.BuildManifest.BuildCommonVoiceManifest_Command import (
    BuildCommonVoiceManifestCommand,
)
from src.slices.BuildManifest.BuildCommonVoiceManifest_Response import (
    BuildCommonVoiceManifestResponse,
)
from src.slices.BuildManifest.SilenceTrim import speech_span

# Common Voice writes its TSVs unquoted; a stray `"` in a sentence must stay a character.
_TSV_FIELD_LIMIT = 1 << 20
# Far above any read-aloud rate (a brisk reader is ~15 chars/s): a transcript this dense for its
# audio is a truncated recording, and CTC over 12.5 Hz frames could not place it anyway.
_MAX_CHARS_PER_S = 25.0
_LOG_EVERY = 50_000
_SECONDS_PER_HOUR = 3600.0

# (uttid, audio_filepath, text)
_Clip = tuple[str, str, str]


def _init_worker() -> None:
    # One intra-op thread per worker, as in PrecomputeFeatures_Handler: the parallelism that pays
    # is across clips, and per-process torch thread pools only spin against each other.
    torch.set_num_threads(1)


def _measure(path: str) -> int | None:
    try:
        return int(load_audio(path).numel()) or None
    except RuntimeError:  # soundfile's LibsndfileError: a corrupt or empty clip
        return None


def _measure_trimmed(path: str) -> tuple[int, int, int] | None:
    try:
        wave = load_audio(path)
    except RuntimeError:
        return None
    if wave.numel() == 0:
        return None
    start, num = speech_span(wave, get_config().audio.sample_rate, get_config().trim)
    return int(wave.numel()), start, num


def _read_tsv(path: str) -> Iterator[dict[str, str]]:
    csv.field_size_limit(_TSV_FIELD_LIMIT)
    with open(path, encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def _clip(clips_dir: str, row: dict[str, str], text: str) -> _Clip:
    return os.path.splitext(row["path"])[0], os.path.join(clips_dir, row["path"]), text


def _published(rows: list[dict[str, str]], clips_dir: str, name: str, log: Any) -> list[_Clip]:
    # Dev and test stay as published (and untrimmed: published Common Voice WERs are on whole
    # clips). Only a sentence the output alphabet cannot spell is dropped, and counted.
    clips = []
    for row in rows:
        text = normalise_transcript(row["sentence"])
        if text is not None:
            clips.append(_clip(clips_dir, row, text))
    log.info(f"cv-{name}: {len(clips):,} of {len(rows):,} clips (unspellable dropped)")
    return clips


def _train_candidates(
    validated: str, clips_dir: str, heldout: list[dict[str, str]], dropped: dict[str, int]
) -> list[_Clip]:
    held_clips = {r["path"] for r in heldout}
    held_speakers = {r["client_id"] for r in heldout}
    held_sentences = {normalise_transcript(r["sentence"]) for r in heldout} - {None}
    clips = []
    for row in _read_tsv(validated):
        if row["path"] in held_clips:
            dropped["heldout_clip"] += 1
            continue
        if row["client_id"] in held_speakers:
            dropped["heldout_speaker"] += 1
            continue
        text = normalise_transcript(row["sentence"])
        if text is None:
            dropped["unspellable"] += 1
            continue
        if text in held_sentences:
            dropped["heldout_sentence"] += 1
            continue
        clips.append(_clip(clips_dir, row, text))
    return clips


def _progress(results: Iterator[Any], total: int, log: Any) -> Iterator[Any]:
    start = time.monotonic()
    for done, result in enumerate(results, 1):
        if done % _LOG_EVERY == 0 or done == total:
            rate = done / max(time.monotonic() - start, 1e-9)
            log.info(f"decoded {done:,}/{total:,} clips ({rate:.0f}/s)")
        yield result


def _row(clip: _Clip, num_samples: int) -> dict[str, Any]:
    uttid, path, text = clip
    return {"uttid": uttid, "audio_filepath": path, "text": text, "num_samples": num_samples}


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as sink:
        for row in sorted(rows, key=lambda r: r["uttid"]):
            sink.write(json.dumps(row) + "\n")


def _strided(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    # Strided over the uttid-sorted rows rather than a head slice, so the subset keeps the full
    # set's speaker and sentence mix and is identical on every rebuild.
    ordered = sorted(rows, key=lambda r: r["uttid"])
    if size >= len(ordered):
        return ordered
    stride = len(ordered) / size
    return [ordered[int(i * stride)] for i in range(size)]


def build_common_voice_manifest(
    cmd: BuildCommonVoiceManifestCommand,
) -> BuildCommonVoiceManifestResponse:
    log = configure_logging()
    clips_dir = os.path.join(cmd.corpus_dir, "clips")
    tsv = {
        name: os.path.join(cmd.corpus_dir, f"{name}.tsv") for name in ("validated", "dev", "test")
    }
    for path in (*tsv.values(), clips_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    dev_tsv, test_tsv = list(_read_tsv(tsv["dev"])), list(_read_tsv(tsv["test"]))
    dropped = dict.fromkeys(
        ("heldout_clip", "heldout_speaker", "unspellable", "heldout_sentence"), 0
    )
    train = _train_candidates(tsv["validated"], clips_dir, dev_tsv + test_tsv, dropped)
    dev = _published(dev_tsv, clips_dir, "dev", log)
    test = _published(test_tsv, clips_dir, "test", log)
    log.info(f"cv-train candidates: {len(train):,} clips; dropped {dropped}")

    with mp.get_context("spawn").Pool(cmd.num_workers, initializer=_init_worker) as pool:
        spans = list(
            _progress(
                pool.imap(_measure_trimmed, [c[1] for c in train], chunksize=64), len(train), log
            )
        )
        lengths = list(pool.imap(_measure, [c[1] for c in dev + test], chunksize=64))
    if any(n is None for n in lengths):
        log.warning(f"cv-dev/test: {lengths.count(None)} undecodable clips dropped")

    sample_rate = get_config().audio.sample_rate
    dropped["undecodable"] = dropped["too_fast"] = 0
    train_rows: list[dict[str, Any]] = []
    untrimmed = kept = 0
    for clip, span in zip(train, spans):
        if span is None:
            dropped["undecodable"] += 1
            continue
        full, start, num = span
        if len(clip[2]) / (num / sample_rate) > _MAX_CHARS_PER_S:
            dropped["too_fast"] += 1
            continue
        row = _row(clip, num)
        if (start, num) != (0, full):
            row["start_sample"] = start
        train_rows.append(row)
        untrimmed, kept = untrimmed + full, kept + num

    def measured(clips: list[_Clip], counts: list[int | None]) -> list[dict[str, Any]]:
        return [_row(c, n) for c, n in zip(clips, counts) if n is not None]

    dev_rows = measured(dev, lengths[: len(dev)])
    test_rows = measured(test, lengths[len(dev) :])
    subset = _strided(dev_rows, cmd.dev_subset_size)
    for path, rows in (
        (cmd.train_out, train_rows),
        (cmd.dev_out, dev_rows),
        (cmd.test_out, test_rows),
        (cmd.dev_subset_out, subset),
    ):
        _write(path, rows)

    response = BuildCommonVoiceManifestResponse(
        train_rows=len(train_rows),
        dev_rows=len(dev_rows),
        test_rows=len(test_rows),
        dev_subset_rows=len(subset),
        dropped=dropped,
        trimmed_rows=sum("start_sample" in r for r in train_rows),
        train_hours_untrimmed=untrimmed / sample_rate / _SECONDS_PER_HOUR,
        train_hours=kept / sample_rate / _SECONDS_PER_HOUR,
    )
    log.info(f"cv manifests written: {response}")
    return response
