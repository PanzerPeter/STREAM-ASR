import json

import soundfile as sf
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Handler import EvaluateCorpus_Handler, subsample
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response


class _StubDecoder:
    # Duck-typed stand-in for StreamingDecoder_Handler: returns a canned hypothesis (and canned
    # timings) per waveform LENGTH, so the aggregation is tested without a model. The handler
    # prefetches and loads audio itself, so the key has to be something the waveform carries.
    def __init__(
        self, hyps: dict[int, str], per_utt: dict[int, tuple[float, float, float]]
    ) -> None:
        self.hyps = hyps
        self.per_utt = per_utt  # samples -> (audio_s, decode_s, finalize_s)

    def decode_waveform(self, wave: torch.Tensor, streaming: bool) -> StreamingDecode_Response:
        n = int(wave.shape[0])
        audio_s, decode_s, finalize_s = self.per_utt[n]
        return StreamingDecode_Response(
            text=self.hyps[n],
            segments=[],
            rtf=decode_s / audio_s,
            decode_s=decode_s,
            audio_s=audio_s,
            finalize_s=finalize_s,
            first_partial_latency_s=0.05 if streaming else None,
        )


def _wav(tmp_path, name: str, n_samples: int) -> str:
    p = tmp_path / f"{name}.wav"
    sf.write(str(p), torch.zeros(n_samples).numpy(), get_config().audio.sample_rate)
    return str(p)


def _manifest(tmp_path, rows: list[dict]) -> str:
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _two_utt_fixture(tmp_path, texts: tuple[str, str], hyps: tuple[str, str]):
    # Utterance A is 1 s of audio decoded in 0.1 s; utterance B is 3 s decoded in 0.9 s.
    sr = get_config().audio.sample_rate
    a, b = _wav(tmp_path, "a", sr), _wav(tmp_path, "b", 3 * sr)
    manifest = _manifest(
        tmp_path,
        [{"audio_filepath": a, "text": texts[0]}, {"audio_filepath": b, "text": texts[1]}],
    )
    dec = _StubDecoder(
        {sr: hyps[0], 3 * sr: hyps[1]},
        {sr: (1.0, 0.1, 0.02), 3 * sr: (3.0, 0.9, 0.06)},
    )
    return manifest, dec


def test_handler_aggregates_wer_and_duration_weighted_rtf(tmp_path):
    manifest, dec = _two_utt_fixture(
        tmp_path, ("THE CAT SAT", "HELLO WORLD"), ("THE CAT SAT", "HELLO WORD")
    )
    resp = EvaluateCorpus_Handler(dec).run(
        EvaluateCorpus_Command(manifest_path=manifest, mode="offline", ablation_stage="beam")
    )
    assert resp.num_utts == 2
    assert abs(resp.wer - 0.2) < 1e-6  # 1 sub over 5 ref words
    assert (resp.num_word_errors, resp.num_ref_words) == (1, 5)
    # RTF is total decode time over total audio (1.0/4.0), NOT the mean of 0.1 and 0.3: a mean of
    # ratios would let a 1 s utterance outvote a 3 s one.
    assert abs(resp.rtf - 0.25) < 1e-9
    assert abs(resp.finalize_s - 0.04) < 1e-9
    assert resp.latency_s is None  # offline emits no partials
    assert resp.stage == "beam" and resp.mode == "offline"


def test_handler_reports_no_timing_when_the_pass_is_contended(tmp_path):
    # The quality pass runs several decodes on one GPU, so its seconds are partly contention.
    # measure_timing=False must blank every timing field rather than publish a flattering number.
    manifest, dec = _two_utt_fixture(tmp_path, ("ONE", "TWO"), ("ONE", "TWO"))
    resp = EvaluateCorpus_Handler(dec).run(
        EvaluateCorpus_Command(
            manifest_path=manifest, mode="streaming", ablation_stage="beam", measure_timing=False
        )
    )
    assert resp.wer == 0.0
    assert resp.rtf is None and resp.latency_s is None and resp.finalize_s is None
    assert resp.audio_s == 4.0  # the corpus size itself is still reported


def test_subsample_strides_instead_of_head_slicing():
    # Manifests are sorted by uttid, so rows[:n] is a couple of speakers. A capped run must stay a
    # picture of the whole split: stride, cover both ends, keep the count exact and deterministic.
    rows = [{"uttid": i} for i in range(100)]
    picked = subsample(rows, 10)
    assert len(picked) == 10
    assert picked[0]["uttid"] == 0 and picked[-1]["uttid"] == 90
    assert picked == subsample(rows, 10)
    assert subsample(rows, None) == rows and subsample(rows, 500) == rows


class _MapTok:
    def __init__(self, m: dict[tuple[int, ...], str]) -> None:
        self.m = m

    def decode(self, ids: list[int]) -> str:
        return self.m[tuple(ids)]


def test_pick_best_weights_selects_lm_favoured_hypothesis():
    # Rescore-mode weight sweep: for a fixed n-best, alpha=0 must pick the acoustic winner and a
    # large-enough alpha must let the LM's preferred (correct) hypothesis win -- proving the sweep
    # ranks by acoustic + alpha*lm and minimises corpus WER over the cached scores (no decoding).
    from src.slices.Decode.StreamingDecode_Response import NbestEntry
    from src.slices.Evaluate.evaluate import _pick_best_weights

    tok = _MapTok({(1,): "ONE WRONG", (2,): "ONE TWO"})
    # id1: acoustic-preferred but wrong (1 sub / 2 words -> WER 0.5); id2: correct, LM-preferred.
    cache = [
        (
            "ONE TWO",
            [
                NbestEntry(ids=[1], acoustic=1.0, lm=-5.0, ilm=0.0),
                NbestEntry(ids=[2], acoustic=0.9, lm=0.0, ilm=0.0),
            ],
        )
    ]
    best, wer = _pick_best_weights(cache, [0.0, 0.5], [0.0], [0.0], tok)
    assert best == (0.5, 0.0, 0.0)
    assert wer[(0.0, 0.0, 0.0)] == 0.5 and wer[(0.5, 0.0, 0.0)] == 0.0


def test_pick_best_weights_uses_ilm_subtraction_to_break_a_tie():
    # beta must actually enter the ranking: with the two hypotheses tied on acoustic + alpha*lm,
    # only subtracting the internal-LM term can move the winner -- the double-count the transducer
    # carries is what ILME removes.
    from src.slices.Decode.StreamingDecode_Response import NbestEntry
    from src.slices.Evaluate.evaluate import _pick_best_weights

    tok = _MapTok({(1,): "ONE WRONG", (2,): "ONE TWO"})
    cache = [
        (
            "ONE TWO",
            [
                NbestEntry(ids=[1], acoustic=1.0, lm=0.0, ilm=0.0),
                NbestEntry(ids=[2], acoustic=1.0, lm=0.0, ilm=-4.0),
            ],
        )
    ]
    best, wer = _pick_best_weights(cache, [0.0], [0.0, 0.5], [0.0], tok)
    assert best == (0.0, 0.5, 0.0)
    assert wer[(0.0, 0.5, 0.0)] == 0.0


def test_oracle_wer_is_the_best_reachable_hypothesis_in_the_nbest():
    # The oracle picks the lowest-error entry per utterance: it bounds what any rescoring weights
    # can reach, so a tuned WER sitting on it means the beam, not the LM, is the bottleneck.
    from src.slices.Decode.StreamingDecode_Response import NbestEntry
    from src.slices.Evaluate.evaluate import _oracle_wer

    tok = _MapTok({(1,): "ONE WRONG", (2,): "ONE TWO"})
    cache = [
        (
            "ONE TWO",
            [
                NbestEntry(ids=[1], acoustic=1.0, lm=0.0, ilm=0.0),
                NbestEntry(ids=[2], acoustic=-9.0, lm=-9.0, ilm=0.0),
            ],
        )
    ]
    assert _oracle_wer(cache, tok) == 0.0


def test_handler_respects_limit(tmp_path):
    manifest, dec = _two_utt_fixture(tmp_path, ("ONE", "TWO"), ("ONE", "TWO"))
    resp = EvaluateCorpus_Handler(dec).run(
        EvaluateCorpus_Command(
            manifest_path=manifest, mode="streaming", ablation_stage="greedy_transducer", limit=1
        )
    )
    assert resp.num_utts == 1
    assert resp.latency_s == 0.05  # streaming DOES have a first partial


def test_stage_table_is_a_cumulative_ablation():
    # The table is the whole definition of what each reported row means: greedy is the beam-1 run,
    # beam widens it, and only the last stage may consult the LM. If a non-LM stage ever set
    # fuse_lm, its row would stop being the acoustic baseline the others are read against.
    from src.slices.Evaluate.DecodePass import STAGES, stage_uses_lm

    assert STAGES["greedy_transducer"].beam_size == 1
    assert STAGES["beam"].beam_size is None  # None = the configured decode.beam_size
    assert STAGES["beam_lm"].beam_size is None
    assert [stage_uses_lm(s) for s in ("greedy_transducer", "beam", "beam_lm")] == [
        False,
        False,
        True,
    ]


def test_each_split_binds_its_own_dev_manifest_and_report_path():
    # --clean/--other pick an acoustic condition, not a file: the dev manifest tuned on has to
    # match the test manifest scored, and the two conditions must not share a report path.
    from src.shared_kernel.Config_Adapter import get_config as _cfg
    from src.slices.Evaluate.evaluate import _SPLITS

    assert set(_SPLITS) == {"clean", "other"}
    for name, split in _SPLITS.items():
        assert split.name == name
        assert ("other" in split.test_manifest) == (name == "other")
        assert ("other" in split.dev_manifest) == (name == "other")
        assert split.test_manifest != split.dev_manifest
    paths = {_cfg().eval.report_path.format(split=n) for n in _SPLITS}
    assert len(paths) == len(_SPLITS)
