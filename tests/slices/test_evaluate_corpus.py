import json

from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Handler import EvaluateCorpus_Handler
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response


class _StubDecoder:
    # Duck-typed stand-in for StreamingDecoder_Handler: returns a canned hypothesis per audio path
    # so the aggregation (WER/CER, mean RTF/latency, limit) is tested without a real model.
    def __init__(self, hyps: dict[str, str]) -> None:
        self.hyps = hyps

    def decode(self, cmd: object) -> StreamingDecode_Response:
        path = cmd.audio_path  # type: ignore[attr-defined]
        return StreamingDecode_Response(
            text=self.hyps[path], segments=[], rtf=0.1, first_partial_latency_s=0.05
        )


def _manifest(tmp_path, rows: list[dict]) -> str:
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_handler_aggregates_wer_and_means(tmp_path):
    manifest = _manifest(
        tmp_path,
        [
            {"audio_filepath": "a", "text": "THE CAT SAT"},
            {"audio_filepath": "b", "text": "HELLO WORLD"},
        ],
    )
    dec = _StubDecoder({"a": "THE CAT SAT", "b": "HELLO WORD"})  # 1 sub over 5 ref words -> 0.2
    resp = EvaluateCorpus_Handler(dec).run(
        EvaluateCorpus_Command(
            manifest_path=manifest, mode="offline", ablation_stage="attn_rescore"
        )
    )
    assert resp.num_utts == 2
    assert abs(resp.wer - 0.2) < 1e-6
    assert abs(resp.rtf - 0.1) < 1e-9
    assert abs(resp.latency_s - 0.05) < 1e-9
    assert resp.stage == "attn_rescore" and resp.mode == "offline"


def test_handler_respects_limit(tmp_path):
    manifest = _manifest(
        tmp_path,
        [{"audio_filepath": "a", "text": "ONE"}, {"audio_filepath": "b", "text": "TWO"}],
    )
    dec = _StubDecoder({"a": "ONE", "b": "TWO"})
    resp = EvaluateCorpus_Handler(dec).run(
        EvaluateCorpus_Command(
            manifest_path=manifest, mode="streaming", ablation_stage="ctc_greedy", limit=1
        )
    )
    assert resp.num_utts == 1
