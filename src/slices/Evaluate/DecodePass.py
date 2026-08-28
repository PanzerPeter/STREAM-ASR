# One decode pass over a manifest, and the stage -> decode-feature table it is configured from.
#
# Lives apart from evaluate.py because a pass is what gets shipped to a worker PROCESS, and spawn
# has to import the entry point by name.
#
# Why processes and not threads: a decode is bounded by the Python bookkeeping of the RNN-T beam
# (recombination, candidate tuples, per-symbol host sync), not by the GPU, and the GIL serialises
# exactly that. Measured on the RTX 5070 over 6 passes x 40 test-clean utterances:
#
#   threads   1 -> 6 workers:  100.7 s -> 89.5 s  (1.1x), GPU pinned at ~30 %
#   processes 1 -> 6 workers:   96.0 s -> 53.5 s  (1.8x), GPU 31 % -> 85 %, VRAM 2.4 -> 5.3 GB
#
# Threads buy nothing; a process has its own interpreter, so the passes genuinely overlap. Returns
# flatten past 3-4 workers (GPU already ~85 %), and the price is a CUDA context plus a model copy
# per worker -- which is why the worker count is bounded by VRAM, not by cores.
from dataclasses import dataclass

import torch

from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecode_Response import NbestEntry
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Handler import EvaluateCorpus_Handler, subsample
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Progress import Progress

# Per-utterance rescore state a tuning pass returns: (reference, acoustic n-best with its terms).
RescoreCache = list[tuple[str, list[NbestEntry]]]


@dataclass(frozen=True)
class StageFlags:
    beam_size: int | None
    fuse_lm: bool


# Cumulative ablation for the single-pass transducer: greedy (beam_size=1, no LM) -> beam (full
# beam_size, no LM) -> beam+LM (full beam, LM n-best rescoring). The lm stage needs lm_weight > 0
# for the scorer to be built at all.
STAGES: dict[str, StageFlags] = {
    "greedy_transducer": StageFlags(1, False),
    "beam": StageFlags(None, False),
    "beam_lm": StageFlags(None, True),
}


def stage_uses_lm(stage: str) -> bool:
    return STAGES[stage].fuse_lm


@dataclass(frozen=True)
class DecodePassJob:
    # Everything a worker process needs to reconstruct the pass from nothing: models are rebuilt in
    # the child, never pickled across.
    kind: str  # "score" (WER/CER table row) | "nbest" (cached n-best for the weight sweep)
    checkpoint: str
    tokenizer: str
    stage: str
    mode: str
    manifest: str
    limit: int | None
    lm_weight: float
    ilm_weight: float
    length_bonus: float
    measure_timing: bool
    label: str


# Per-process caches: a worker that handles two jobs must not reload a 220 MB checkpoint or a
# 350 MB LM to do it, and the parent reuses the same entries for its serial timing pass.
_MODELS: dict[tuple[str, str], tuple[TransducerModel, SentencePieceTokenizer]] = {}
_DECODERS: dict[tuple[str, str, str, float, float, float], StreamingDecoder_Handler] = {}


def _model_and_tok(job: DecodePassJob) -> tuple[TransducerModel, SentencePieceTokenizer]:
    key = (job.checkpoint, job.tokenizer)
    if key not in _MODELS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TransducerModel()
        load_checkpoint(job.checkpoint, model)
        _MODELS[key] = (model.to(device).eval(), SentencePieceTokenizer(job.tokenizer))
    return _MODELS[key]


def build_decoder(
    model: TransducerModel,
    tok: SentencePieceTokenizer,
    stage: str,
    lm_weight: float,
    ilm_weight: float,
    length_bonus: float,
) -> StreamingDecoder_Handler:
    # The decoder loads the LM only when the stage's fuse_lm gate is set AND lm_weight > 0, so
    # non-LM stages (and alpha == 0) pay no LM cost; that same gate zeroes beta, keeping the
    # acoustic-only stages byte-identical to a run without ILME.
    f = STAGES[stage]
    return StreamingDecoder_Handler(
        model,
        tok,
        beam_size=f.beam_size,
        fuse_lm=f.fuse_lm,
        lm_weight=lm_weight,
        ilm_weight=ilm_weight,
        length_bonus=length_bonus,
    )


def _decoder_for(job: DecodePassJob) -> StreamingDecoder_Handler:
    key = (
        job.checkpoint,
        job.tokenizer,
        job.stage,
        job.lm_weight,
        job.ilm_weight,
        job.length_bonus,
    )
    if key not in _DECODERS:
        model, tok = _model_and_tok(job)
        _DECODERS[key] = build_decoder(
            model, tok, job.stage, job.lm_weight, job.ilm_weight, job.length_bonus
        )
    return _DECODERS[key]


def _score(job: DecodePassJob, decoder: StreamingDecoder_Handler) -> EvaluateCorpus_Response:
    return EvaluateCorpus_Handler(decoder, label=job.label).run(
        EvaluateCorpus_Command(
            manifest_path=job.manifest,
            mode=job.mode,
            ablation_stage=job.stage,
            limit=job.limit,
            measure_timing=job.measure_timing,
        )
    )


def _nbest(job: DecodePassJob, decoder: StreamingDecoder_Handler) -> RescoreCache:
    # Acoustic n-best with the external-LM and internal-LM sequence terms kept apart, so the caller
    # can rank at any (alpha, beta) without decoding again.
    rows = subsample(load_manifest(job.manifest), job.limit)
    prog = Progress(job.label, len(rows))
    cache: RescoreCache = []
    for i, r in enumerate(rows, 1):
        cache.append(
            (
                r["text"],
                decoder.nbest_for_rescore(
                    StreamingDecode_Command(
                        audio_path=r["audio_filepath"], streaming=job.mode == "streaming"
                    )
                ),
            )
        )
        prog.tick(i)
    prog.done()
    return cache


def run_job(job: DecodePassJob) -> EvaluateCorpus_Response | RescoreCache:
    # Process entry point. no_grad is thread-local, so it is entered here rather than by the caller.
    with torch.no_grad():
        decoder = _decoder_for(job)
        return _score(job, decoder) if job.kind == "score" else _nbest(job, decoder)
