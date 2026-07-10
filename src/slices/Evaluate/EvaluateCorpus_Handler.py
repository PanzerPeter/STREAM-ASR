# src/slices/Evaluate/EvaluateCorpus_Handler.py
# Corpus evaluation driver: runs the Decode slice's handler per utterance and aggregates WER/CER
# (jiwer) + mean RTF/latency. The decoder is pre-configured for its ablation stage by the caller,
# so this harness stays stage-agnostic (one loop produces every row of the ablation table).
from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Metrics import corpus_wer, corpus_cer


class EvaluateCorpus_Handler:
    def __init__(self, decoder: StreamingDecoder_Handler) -> None:
        self.decoder = decoder

    def run(self, cmd: EvaluateCorpus_Command) -> EvaluateCorpus_Response:
        rows = load_manifest(cmd.manifest_path)
        if cmd.limit is not None:
            rows = rows[: cmd.limit]
        streaming = cmd.mode == "streaming"
        refs: list[str] = []
        hyps: list[str] = []
        rtfs: list[float] = []
        lats: list[float] = []
        for r in rows:
            resp = self.decoder.decode(
                StreamingDecode_Command(audio_path=r["audio_filepath"], streaming=streaming)
            )
            refs.append(r["text"])
            hyps.append(resp.text)
            rtfs.append(resp.rtf)
            lats.append(resp.first_partial_latency_s)
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
