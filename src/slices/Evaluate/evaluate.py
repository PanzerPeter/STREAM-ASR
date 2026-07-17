# src/slices/Evaluate/evaluate.py — corpus WER/CER/RTF/latency + ablation table (GPU; user-run).
# Loops the configured ablation stages x {offline, streaming}, building a decoder configured for
# each stage, and writes the full report to EvalConfig.report_path. Model/tokenizer loading mirrors
# src/slices/Decode/streaming_decode.py.
#
# LM evaluation: the LM contributes at alpha (lm_weight) > 0 by rescoring the acoustic n-best. To
# evaluate it *honestly* the weight is tuned on a dev manifest and frozen for the test run. `--tune
# DEV` does that in one command: it decodes DEV once acoustic-only, sweeps `--lm-grid` over the
# cached (acoustic, LM-sequence) scores for free, picks the WER-minimising alpha, then runs the
# full test table with it. Without `--tune`, a fixed alpha comes from `--lm-weight` (or
# decode.lm_weight); alpha == 0 reproduces the pure-acoustic decoder exactly and LM stages inactive.
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch

from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Handler import EvaluateCorpus_Handler
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Metrics import corpus_wer

_T = TypeVar("_T")
# Two decode runs share the GPU concurrently: decoding is largely sequential per utterance, so a
# second run overlaps the first's Python/copy gaps. Inference is read-only on the shared model, so
# the only real hazard is grad state — each task re-enters torch.no_grad() (thread-local, not
# inherited from the launching thread).
_MAX_PARALLEL = 2


def _map_parallel(fn: Callable[..., _T], tasks: list[tuple]) -> list[_T]:
    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        return [f.result() for f in [pool.submit(fn, *t) for t in tasks]]


@dataclass(frozen=True)
class _StageFlags:
    beam_size: int | None
    fuse_lm: bool


# Cumulative ablation for the single-pass transducer: greedy (beam_size=1, no LM) -> beam (full
# beam_size, no LM) -> beam+LM (full beam, LM shallow fusion). Replaces the old two-pass ladder
# (prefix beam / attention rescore / LM rescore) now that Task 10 collapsed decode to one pass.
# The lm stage requires decode.lm_weight > 0 so the scorer is actually built.
_STAGES: dict[str, _StageFlags] = {
    "greedy_transducer": _StageFlags(1, False),
    "beam": _StageFlags(None, False),
    "beam_lm": _StageFlags(None, True),
}


def _stage_uses_lm(stage: str) -> bool:
    return _STAGES[stage].fuse_lm


def _run_stage(
    model: TransducerModel,
    tok: SentencePieceTokenizer,
    stage: str,
    lm_weight: float,
    manifest: str,
    mode: str,
    limit: int | None,
    label: str = "",
) -> EvaluateCorpus_Response:
    # Build a decoder configured for exactly this ablation stage + alpha, then score the corpus.
    # The decoder loads the LM only when the stage's fuse_lm gate is set AND lm_weight > 0, so
    # non-LM stages (and alpha == 0) pay no LM cost.
    f = _STAGES[stage]
    decoder = StreamingDecoder_Handler(
        model,
        tok,
        beam_size=f.beam_size,
        fuse_lm=f.fuse_lm,
        lm_weight=lm_weight,
    )
    handler = EvaluateCorpus_Handler(decoder, label=label or f"{stage}/{mode}")
    # no_grad is thread-local: set it here so this holds whether _run_stage runs inline (tuning) or
    # inside a worker thread (the parallel table).
    with torch.no_grad():
        return handler.run(
            EvaluateCorpus_Command(
                manifest_path=manifest, mode=mode, ablation_stage=stage, limit=limit
            )
        )


# Cached per-utterance rescore state: (reference text, n-best of (ids, acoustic_logp, lm_seq_logp)).
_RescoreCache = list[tuple[str, list[tuple[list[int], float, float]]]]


def _pick_best_alpha(
    cache: _RescoreCache, grid: list[float], tok: SentencePieceTokenizer
) -> tuple[float, dict[float, float]]:
    # Pure alpha sweep over cached scores -- no decoding. For each alpha, each utterance emits the
    # n-best hypothesis maximising acoustic + alpha*lm; corpus WER over those picks scores it.
    # Returns the WER-minimising alpha and the full alpha->WER map (for logging). An empty n-best
    # (silence / no emission) contributes an empty hypothesis, exactly as the decode path would.
    wer_by_alpha: dict[float, float] = {}
    refs = [ref for ref, _ in cache]
    for alpha in grid:
        hyps = [
            tok.decode(max(nb, key=lambda h: h[1] + alpha * h[2])[0]) if nb else ""
            for _, nb in cache
        ]
        wer_by_alpha[alpha] = corpus_wer(refs, hyps)
    best_alpha = min(grid, key=lambda a: wer_by_alpha[a])
    return best_alpha, wer_by_alpha


def _tune_alpha_rescore(
    model: TransducerModel,
    tok: SentencePieceTokenizer,
    dev_manifest: str,
    grid: list[float],
    limit: int | None,
    beam_size: int,
) -> float:
    # Rescore-mode tuning: decode dev ONCE acoustic-only (LM off), cache each utterance's n-best
    # with separated acoustic + unweighted-LM scores, then sweep alpha over the cache for free. This
    # is ~beam_size * n_alpha times cheaper than the shallow-fusion sweep (which re-runs a full
    # beam+LM decode per alpha). The chosen alpha is still applied via true shallow fusion in the
    # final table -- only alpha *selection* is done by rescoring here.
    print(f"--- tuning lm_weight on {dev_manifest} (rescore mode, offline, n<= {limit}) ---")
    print(
        f"    decode dev ONCE acoustic-only, then sweep {len(grid)} alphas over cached LM sequence "
        f"scores (no re-decode).",
        flush=True,
    )
    rescorer = StreamingDecoder_Handler(
        model, tok, beam_size=beam_size, fuse_lm=True, lm_weight=1.0
    )
    rows = load_manifest(dev_manifest)
    if limit is not None:
        rows = rows[:limit]
    cache: _RescoreCache = []
    total = len(rows)
    t0 = time.perf_counter()
    beat = max(1, total // 20)
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            nb = rescorer.nbest_for_rescore(
                StreamingDecode_Command(audio_path=r["audio_filepath"], streaming=False)
            )
            cache.append((r["text"], nb))
            if i == 1 or i % beat == 0 or i == total:
                el = time.perf_counter() - t0
                per = el / i
                print(
                    f"  [rescore-decode] {i}/{total} utts  {el:.0f}s  {per:.2f}s/utt  "
                    f"ETA {per * (total - i):.0f}s",
                    flush=True,
                )
    best_alpha, wer_by_alpha = _pick_best_alpha(cache, grid, tok)
    for alpha in grid:
        marker = "  <- best" if alpha == best_alpha else ""
        print(f"  lm_weight={alpha:<5} dev WER={wer_by_alpha[alpha]:.4f}{marker}", flush=True)
    print(f"--- selected lm_weight={best_alpha} (dev WER={wer_by_alpha[best_alpha]:.4f}) ---")
    return best_alpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--checkpoint", default="data/checkpoints/transducer_best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe500.model")
    ap.add_argument("--limit", type=int, default=None, help="cap utterances in the final table")
    ap.add_argument(
        "--lm-weight",
        type=float,
        default=None,
        help="fixed fusion weight (alpha) for this run; ignored when --tune sweeps one instead",
    )
    ap.add_argument(
        "--tune",
        metavar="DEV_MANIFEST",
        default=None,
        help="sweep --lm-grid on this dev manifest and freeze the best alpha for the final table",
    )
    ap.add_argument(
        "--lm-grid",
        default="0.0,0.1,0.2,0.3,0.4,0.5",
        help="comma-separated alpha grid for --tune",
    )
    ap.add_argument(
        "--tune-limit",
        type=int,
        default=500,
        help="cap dev utterances during --tune (a subset keeps the sweep quick)",
    )
    args = ap.parse_args()

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransducerModel()
    load_checkpoint(args.checkpoint, model)
    model = model.to(device).eval()
    tok = SentencePieceTokenizer(args.tokenizer)

    # _run_stage sets no_grad per task (thread-local), so no outer no_grad wrapper is needed.
    if args.tune is not None:
        grid = [float(x) for x in args.lm_grid.split(",")]
        lm_weight = _tune_alpha_rescore(
            model, tok, args.tune, grid, args.tune_limit, cfg.decode.beam_size
        )
    else:
        # CLI override wins, else the configured value. The authoritative decode.yaml keeps
        # lm_weight=0.0 (the alpha=0 regression lock), so sweeps never mutate it.
        lm_weight = args.lm_weight if args.lm_weight is not None else cfg.decode.lm_weight
    print(f"lm_weight (alpha) = {lm_weight}")

    report = []
    for stage in cfg.eval.ablation_stages:
        if _stage_uses_lm(stage) and lm_weight <= 0:
            # Otherwise this stage silently equals the pure-acoustic beam and the report misleads.
            print(
                f"WARNING: stage '{stage}' uses the LM but lm_weight={lm_weight} -> LM "
                f"inactive; pass --tune DEV or --lm-weight > 0 to evaluate it"
            )
        # offline + streaming for this stage run as the two parallel tasks.
        resps = _map_parallel(
            _run_stage,
            [
                (model, tok, stage, lm_weight, args.manifest, mode, args.limit)
                for mode in ("offline", "streaming")
            ],
        )
        for resp in resps:
            report.append(asdict(resp))
            print(
                f"{resp.stage:<13} {resp.mode:<9} WER={resp.wer:.4f} CER={resp.cer:.4f} "
                f"RTF={resp.rtf:.3f} lat={resp.latency_s:.3f}s n={resp.num_utts}"
            )

    out = Path(cfg.eval.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lm_weight": lm_weight, "rows": report}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
