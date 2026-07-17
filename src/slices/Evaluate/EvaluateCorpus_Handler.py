# src/slices/Evaluate/EvaluateCorpus_Handler.py
# Corpus evaluation driver: runs the Decode slice's handler per utterance and aggregates WER/CER
# (jiwer) + mean RTF/latency. The decoder is pre-configured for its ablation stage by the caller,
# so this harness stays stage-agnostic (one loop produces every row of the ablation table).
import time

from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Metrics import corpus_wer, corpus_cer


class EvaluateCorpus_Handler:
    def __init__(self, decoder: StreamingDecoder_Handler, label: str = "") -> None:
        self.decoder = decoder
        # Free-text tag prefixing the per-utterance heartbeat so parallel stages/alphas stay
        # distinguishable in the interleaved log; falls back to stage/mode when the caller omits it.
        self.label = label

    def run(self, cmd: EvaluateCorpus_Command) -> EvaluateCorpus_Response:
        rows = load_manifest(cmd.manifest_path)
        if cmd.limit is not None:
            rows = rows[: cmd.limit]
        streaming = cmd.mode == "streaming"
        refs: list[str] = []
        hyps: list[str] = []
        rtfs: list[float] = []
        lats: list[float] = []
        # Per-utterance heartbeat: RNN-T beam (esp. beam+LM) is a long silent grind, so log the
        # first utterance (proves the pipeline is live) then ~20 evenly-spaced ticks with
        # elapsed/ETA + running per-utt cost -- the fastest way to see a slow LM stage as itself.
        total = len(rows)
        tag = self.label or f"{cmd.ablation_stage}/{cmd.mode}"
        t0 = time.perf_counter()
        beat = max(1, total // 20)
        for i, r in enumerate(rows, 1):
            resp = self.decoder.decode(
                StreamingDecode_Command(audio_path=r["audio_filepath"], streaming=streaming)
            )
            refs.append(r["text"])
            hyps.append(resp.text)
            rtfs.append(resp.rtf)
            lats.append(resp.first_partial_latency_s)
            if i == 1 or i % beat == 0 or i == total:
                el = time.perf_counter() - t0
                per = el / i
                print(
                    f"  [{tag}] {i}/{total} utts  {el:.0f}s  {per:.2f}s/utt  "
                    f"ETA {per * (total - i):.0f}s  meanRTF={sum(rtfs) / i:.2f}",
                    flush=True,
                )
        n = max(1, len(rows))
        return EvaluateCorpus_Response(
            stage=cmd.ablation_stage,
            mode=cmd.mode,
            wer=corpus_wer(refs, hyps) if refs else 0.0,
            cer=corpus_cer(refs, hyps) if refs else 0.0,
            rtf=sum(rtfs) / n,
            latency_s=sum(lats) / n,
            num_utts=len(rows),
        )
