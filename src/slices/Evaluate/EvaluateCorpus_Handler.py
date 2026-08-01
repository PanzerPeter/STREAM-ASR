# Corpus evaluation driver: runs the Decode slice's handler per utterance and aggregates WER/CER
# (jiwer) + timing. The decoder is pre-configured for its ablation stage by the caller, so this
# harness stays stage-agnostic (one loop produces every row of the ablation table).
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

import torch

from src.shared_kernel.AudioIO_Adapter import load_audio, load_manifest
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Metrics import char_errors, word_errors
from src.slices.Evaluate.Progress import Progress

# How many utterances' audio to decode from FLAC ahead of the model. Loading is libsndfile +
# resample on the CPU and the GPU has nothing to do during it, so one worker running one utterance
# ahead takes it off the critical path entirely; more than that only buys queue depth we never use.
_PREFETCH = 2


class _Decoder(Protocol):
    # Structural type: this harness needs a waveform-in decode and nothing else, so it type-checks
    # against a test stub as readily as against StreamingDecoder_Handler.
    def decode_waveform(self, wave: torch.Tensor, streaming: bool) -> Any: ...


def subsample(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    # An evenly strided subsample, NOT rows[:limit]. Manifests are sorted by uttid, so a head slice
    # is a few speakers reading a few chapters -- a capped run would report that pair's WER, not the
    # split's. Striding keeps the speaker/duration mix of the full manifest and stays deterministic.
    if limit is None or limit >= len(rows) or limit <= 0:
        return rows
    stride = len(rows) / limit
    return [rows[min(len(rows) - 1, int(i * stride))] for i in range(limit)]


class EvaluateCorpus_Handler:
    def __init__(self, decoder: _Decoder, label: str = "") -> None:
        self.decoder = decoder
        # Free-text tag prefixing the per-utterance heartbeat so parallel stages stay
        # distinguishable in the interleaved log; falls back to stage/mode when the caller omits it.
        self.label = label

    def run(self, cmd: EvaluateCorpus_Command) -> EvaluateCorpus_Response:
        rows = subsample(load_manifest(cmd.manifest_path), cmd.limit)
        streaming = cmd.mode == "streaming"
        prog = Progress(self.label or f"{cmd.ablation_stage}/{cmd.mode}", len(rows))
        w_err = w_ref = c_err = c_ref = 0
        audio_s = decode_s = finalize_s = 0.0
        lats: list[float] = []

        # Audio IO runs one utterance ahead of the model on its own thread; the decode timings the
        # response carries start after the waveform is in hand, so prefetching cannot flatter them.
        io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-io")
        queued: deque[Future[torch.Tensor]] = deque()
        try:
            for r in rows[:_PREFETCH]:
                queued.append(io.submit(load_audio, r["audio_filepath"]))
            for i, r in enumerate(rows, 1):
                wave = queued.popleft().result()
                ahead = i - 1 + _PREFETCH
                if ahead < len(rows):
                    queued.append(io.submit(load_audio, rows[ahead]["audio_filepath"]))
                resp = self.decoder.decode_waveform(wave, streaming)
                we, wn = word_errors(r["text"], resp.text)
                ce, cn = char_errors(r["text"], resp.text)
                w_err, w_ref, c_err, c_ref = w_err + we, w_ref + wn, c_err + ce, c_ref + cn
                audio_s += resp.audio_s
                decode_s += resp.decode_s
                finalize_s += resp.finalize_s
                if resp.first_partial_latency_s is not None:
                    lats.append(resp.first_partial_latency_s)
                prog.tick(i, f"WER={100 * w_err / max(1, w_ref):.2f}%")
        finally:
            io.shutdown(wait=True)
        prog.done(f"WER={100 * w_err / max(1, w_ref):.2f}%")

        timed = cmd.measure_timing
        return EvaluateCorpus_Response(
            stage=cmd.ablation_stage,
            mode=cmd.mode,
            wer=w_err / max(1, w_ref),
            cer=c_err / max(1, c_ref),
            num_utts=len(rows),
            num_word_errors=w_err,
            num_ref_words=w_ref,
            num_char_errors=c_err,
            num_ref_chars=c_ref,
            audio_s=audio_s,
            # Duration-weighted: the only definition that survives LibriSpeech's 1.3-35 s spread.
            rtf=(decode_s / max(audio_s, 1e-6)) if timed else None,
            latency_s=(sum(lats) / len(lats)) if (timed and lats) else None,
            finalize_s=(finalize_s / len(rows)) if (timed and rows) else None,
        )
