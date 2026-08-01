# Corpus WER/CER/RTF/latency + ablation table (GPU; user-run).
#
# `--clean` / `--other` (exactly one, no default) name a LibriSpeech condition, not a file: each
# binds BOTH the test manifest that is scored and the dev manifest the rescoring weights are tuned
# on, so a test-other table can never be reported against weights tuned on dev-clean, and the two
# reports cannot land on the same path.
#
# The run has two passes, deliberately separated:
#   quality  -- every ablation stage x {offline, streaming} decoded CONCURRENTLY (in worker
#               processes -- see DecodePass) over the full manifest. WER/CER do not care who else
#               is on the card, so this pass is free to buy wall clock with contention.
#   timing   -- the same configurations re-run ALONE and serially over a strided subsample. RTF,
#               first-partial latency and finalize cost are only meaningful without contention, so
#               they are measured here and nowhere else.
#
# LM evaluation: the LM contributes at alpha (lm_weight) > 0 by rescoring the acoustic n-best. To
# evaluate it *honestly* the weights are tuned on the split's dev manifest and frozen for the test
# run -- automatic unless --no-tune or an explicit --lm-weight/--ilm-weight is given. Tuning decodes
# dev once PER MODE acoustic-only, then sweeps --lm-grid x --ilm-grid over the cached
# (acoustic, LM-sequence, internal-LM-sequence) scores for free: neither weight moves the acoustic
# beam, so no grid point costs another decode. Offline and streaming get their own (alpha, beta) --
# streaming's acoustic scores are weaker, and the weight that best offsets them is not the offline
# one.
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, cast

import torch

from src.shared_kernel.AudioIO_Adapter import load_manifest
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.Decode.StreamingDecode_Response import NbestEntry
from src.slices.Evaluate.DecodePass import (
    STAGES,
    DecodePassJob,
    RescoreCache,
    run_job,
    stage_uses_lm,
)
from src.slices.Evaluate.EvaluateCorpus_Handler import subsample
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response
from src.slices.Evaluate.Metrics import word_errors
from src.slices.Evaluate.Progress import log

_MODES = ("offline", "streaming")


@dataclass(frozen=True)
class _Split:
    # A LibriSpeech acoustic condition: the test manifest to score and the dev manifest to tune on.
    # They must move together -- dev-clean weights applied to test-other is tuning on the wrong
    # acoustics, and one shared report path lets the second run silently overwrite the first.
    name: str
    test_manifest: str
    dev_manifest: str


_SPLITS = {
    "clean": _Split("clean", "data/manifests/test.jsonl", "data/manifests/dev.jsonl"),
    "other": _Split("other", "data/manifests/test-other.jsonl", "data/manifests/dev-other.jsonl"),
}


def _dispatch(jobs: list[DecodePassJob], workers: int) -> list[object]:
    # Worker PROCESSES, not threads: see the header of DecodePass for the measurement that decided
    # it. One job per process at a time, each rebuilding its own model, so the only shared state is
    # the read-only checkpoint on disk. workers <= 1 runs inline -- no spawn cost, and the serial
    # timing pass needs exactly that path.
    if workers <= 1 or len(jobs) == 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")
    ) as pool:
        return list(pool.map(run_job, jobs))


# Per utterance: the n-best, each entry's word-error count against the reference, and the reference
# word count. Alignment happens ONCE here; a grid point is then an argmax plus a table lookup.
_Prepared = list[tuple[list[NbestEntry], list[int], int]]


class _Tok(Protocol):
    # Structural stand-in for the tokenizer: the sweep only needs `.decode(ids) -> str`.
    def decode(self, ids: list[int]) -> str: ...


def _prepare(cache: RescoreCache, tok: _Tok) -> _Prepared:
    out: _Prepared = []
    for ref, nb in cache:
        # An empty n-best still has a cost -- the whole reference is deleted -- so it gets a
        # single synthetic entry rather than being dropped from the corpus divisor.
        texts = [tok.decode(h.ids) for h in nb] or [""]
        stats = [word_errors(ref, t) for t in texts]
        out.append((nb, [s[0] for s in stats], stats[0][1]))
    return out


def _sweep(
    prepared: _Prepared, alpha_grid: list[float], beta_grid: list[float], length_bonus: float
) -> dict[tuple[float, float], float]:
    # Pure (alpha, beta) sweep over cached scores -- no decoding, no re-alignment. For each point,
    # each utterance emits the n-best hypothesis maximising
    #     acoustic + alpha*lm - beta*ilm + length_bonus*len
    # and the corpus WER over those picks scores the point. The ranking expression must match the
    # live decode one (StreamingDecoder_Handler._search_rescore) so the weights chosen here are the
    # weights used there.
    total_words = max(1, sum(p[2] for p in prepared))
    wer_by_point: dict[tuple[float, float], float] = {}
    for alpha in alpha_grid:
        for beta in beta_grid:
            errors = 0
            for nb, errs, _ in prepared:
                # First-max wins ties, matching the live rescorer: its sort is stable, so tied
                # hypotheses keep the acoustic beam's order there too.
                best_i, best_score = 0, float("-inf")
                for i, h in enumerate(nb):
                    score = h.acoustic + alpha * h.lm - beta * h.ilm + length_bonus * len(h.ids)
                    if score > best_score:
                        best_i, best_score = i, score
                errors += errs[best_i]
            wer_by_point[(alpha, beta)] = errors / total_words
    return wer_by_point


def _oracle_from(prepared: _Prepared) -> float:
    # Lower bound on what ANY rescorer can reach: pick, per utterance, the n-best entry with the
    # fewest word errors against its own reference. If the tuned WER already sits near this, the
    # bottleneck is the acoustic beam's coverage (widen it), not the rescoring weights or the LM.
    total_words = max(1, sum(p[2] for p in prepared))
    return sum(min(errs) for _, errs, _ in prepared) / total_words


def _pick_best_weights(
    cache: RescoreCache,
    alpha_grid: list[float],
    beta_grid: list[float],
    tok: _Tok,
    length_bonus: float = 0.0,
) -> tuple[tuple[float, float], dict[tuple[float, float], float]]:
    wer_by_point = _sweep(_prepare(cache, tok), alpha_grid, beta_grid, length_bonus)
    return min(wer_by_point, key=lambda p: wer_by_point[p]), wer_by_point


def _oracle_wer(cache: RescoreCache, tok: _Tok) -> float:
    return _oracle_from(_prepare(cache, tok))


def _tune_rescore_weights(
    tok: _Tok,
    checkpoint: str,
    tokenizer_path: str,
    dev_manifest: str,
    alpha_grid: list[float],
    beta_grid: list[float],
    limit: int | None,
    workers: int,
) -> dict[str, dict[str, float | None]]:
    # Rescore-mode tuning, per mode: decode dev ONCE acoustic-only (LM off), cache each utterance's
    # n-best with its acoustic / external-LM / internal-LM terms kept apart, then sweep the whole
    # (alpha, beta) grid over the cache for free. The two modes' dev decodes are independent, so
    # they run as two worker processes exactly like the quality pass.
    rows = subsample(load_manifest(dev_manifest), limit)
    log(
        f"--- tuning (alpha, beta) on {dev_manifest}: n={len(rows)}, per mode, "
        f"{len(alpha_grid)}x{len(beta_grid)} grid points swept over ONE cached decode each"
    )
    length_bonus = get_config().decode.length_bonus
    # lm_weight=1.0 only has to make the worker BUILD an LM: the job returns raw, unweighted LM and
    # ILM terms, so this weight never enters a score. The searcher is the full-beam one either way.
    jobs = [
        DecodePassJob(
            kind="nbest",
            checkpoint=checkpoint,
            tokenizer=tokenizer_path,
            stage="beam_lm",
            mode=mode,
            manifest=dev_manifest,
            limit=limit,
            lm_weight=1.0,
            ilm_weight=0.0,
            measure_timing=False,
            label=f"tune-decode/{mode}",
        )
        for mode in _MODES
    ]
    caches = [cast(RescoreCache, c) for c in _dispatch(jobs, workers)]

    tuned: dict[str, dict[str, float | None]] = {}
    for mode, cache in zip(_MODES, caches):
        prepared = _prepare(cache, tok)
        wer_by_point = _sweep(prepared, alpha_grid, beta_grid, length_bonus)
        best = min(wer_by_point, key=lambda p: wer_by_point[p])
        acoustic = wer_by_point.get((0.0, 0.0))
        log(f"--- {mode} dev sweep (n={len(rows)}) ---")
        for point in sorted(wer_by_point):
            mark = "  <- best" if point == best else ""
            log(
                f"    alpha={point[0]:<5} beta={point[1]:<5} "
                f"dev WER={100 * wer_by_point[point]:6.2f}%{mark}"
            )
        oracle = _oracle_from(prepared)
        log(
            f"    selected alpha={best[0]} beta={best[1]}  dev WER={100 * wer_by_point[best]:.2f}%"
            + (f" (acoustic-only {100 * acoustic:.2f}%)" if acoustic is not None else "")
            + f" | n-best oracle {100 * oracle:.2f}% -- the floor this beam can be rescored to"
        )
        tuned[mode] = {
            "lm_weight": best[0],
            "ilm_weight": best[1],
            "dev_wer": wer_by_point[best],
            "dev_wer_acoustic": acoustic,  # None unless (0, 0) was on the grid
            "dev_oracle_wer": oracle,
            "dev_utts": float(len(rows)),
        }
    return tuned


def _print_table(rows: list[EvaluateCorpus_Response]) -> None:
    # "-" wherever a number was not measured: an unmeasured latency must not read as a fast one.
    def cell(value: float | None, fmt: str, unit: str = "", scale: float = 1.0) -> str:
        return ("-" if value is None else f"{value * scale:{fmt}}{unit}").rjust(10)

    log("")
    log(
        f"{'stage':<18}{'mode':<10}{'WER':>8}{'CER':>8}{'RTF':>10}{'finalize':>10}"
        f"{'partial':>10}{'n':>7}   word errors"
    )
    log("-" * 95)
    for r in rows:
        log(
            f"{r.stage:<18}{r.mode:<10}{100 * r.wer:>7.2f}%{100 * r.cer:>7.2f}%"
            f"{cell(r.rtf, '.3f')}{cell(r.finalize_s, '.3f', 's')}"
            f"{cell(r.latency_s, '.0f', 'ms', 1000.0)}"
            f"{r.num_utts:>7}   {r.num_word_errors}/{r.num_ref_words}"
        )
    log("")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Corpus WER/CER/RTF ablation for the streaming transducer."
    )
    split = ap.add_mutually_exclusive_group(required=True)
    # No default on purpose: which acoustic condition a WER belongs to is the single most
    # load-bearing fact about it, and it decides the dev manifest and the report path too.
    split.add_argument(
        "--clean",
        dest="split",
        action="store_const",
        const="clean",
        help="test-clean, tuned on dev-clean",
    )
    split.add_argument(
        "--other",
        dest="split",
        action="store_const",
        const="other",
        help="test-other, tuned on dev-other",
    )
    # The averaged tail, not transducer_best.pt: `_best` is one step picked by the ~1k-word dev
    # probe, so part of its lead over its neighbours is sampling noise. Run
    # scripts/average_checkpoints.py after training.
    ap.add_argument("--checkpoint", default="data/checkpoints/transducer_avg.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe500.model")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap test utterances (evenly strided, not a head slice) -- smoke runs only",
    )
    ap.add_argument(
        "--lm-weight",
        type=float,
        default=None,
        help="fixed alpha for both modes; giving it skips the dev sweep",
    )
    ap.add_argument(
        "--ilm-weight",
        type=float,
        default=None,
        help="fixed beta (ILME subtraction) for both modes; giving it skips the dev sweep",
    )
    ap.add_argument(
        "--no-tune",
        action="store_true",
        help="skip the dev sweep entirely and use --lm-weight/--ilm-weight (or decode.yaml)",
    )
    ap.add_argument(
        "--lm-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6", help="alpha grid for the dev sweep"
    )
    ap.add_argument("--ilm-grid", default="0.0,0.1,0.2,0.3", help="beta grid for the dev sweep")
    ap.add_argument(
        "--tune-limit",
        type=int,
        default=None,
        help="cap dev utterances during tuning (strided); default is the WHOLE dev split",
    )
    ap.add_argument(
        "--stages",
        default=None,
        help="comma-separated ablation stages, overriding eval.ablation_stages; "
        "'greedy_transducer,beam' skips the LM stage (which without an LM only repeats the beam)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="concurrent decode passes in the quality pass (default eval.workers). Timing is "
        "measured in a separate serial pass, so this trades nothing but VRAM for wall clock",
    )
    ap.add_argument(
        "--rtf-probe",
        type=int,
        default=None,
        help="utterances in the serial timing probe (default eval.rtf_probe_utts; 0 skips it and "
        "reports null RTF/latency)",
    )
    ap.add_argument("--report", default=None, help="report path; default eval.report_path")
    args = ap.parse_args()

    cfg = get_config()
    sp = _SPLITS[args.split]
    workers = args.workers if args.workers is not None else cfg.eval.workers
    probe_n = args.rtf_probe if args.rtf_probe is not None else cfg.eval.rtf_probe_utts
    stages = args.stages.split(",") if args.stages else list(cfg.eval.ablation_stages)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {sorted(STAGES)}")

    # The parent never loads the model: every decode pass runs in a worker that builds its own, and
    # the serial timing pass reuses one of those workers' path in-process. Only the tokenizer is
    # needed here, to turn cached n-best ids back into text during the weight sweep.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = SentencePieceTokenizer(args.tokenizer)
    n_test = len(subsample(load_manifest(sp.test_manifest), args.limit))

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    log(f"=== evaluate: LibriSpeech {sp.name} =========================================")
    log(f"  test      {sp.test_manifest}  (n={n_test})")
    log(f"  dev       {sp.dev_manifest}")
    log(f"  model     {args.checkpoint}  |  tokenizer {args.tokenizer}")
    log(f"  device    {device} ({gpu})")
    log(
        f"  decode    beam={cfg.decode.beam_size} chunk={cfg.decode.chunk_size} "
        f"max_symbols={cfg.decode.max_symbols} length_bonus={cfg.decode.length_bonus} "
        f"cuda_graph={cfg.decode.cuda_graph}"
    )
    log(f"  stages    {', '.join(stages)} x {', '.join(_MODES)}")
    log(f"  passes    quality: {workers} worker process(es) | timing: serial, n={probe_n}")
    log("=" * 78)

    t_start = time.perf_counter()
    fixed = args.no_tune or args.lm_weight is not None or args.ilm_weight is not None
    tuning: dict[str, dict[str, float | None]] = {}
    if any(stage_uses_lm(s) for s in stages) and not fixed:
        tuning = _tune_rescore_weights(
            tok,
            args.checkpoint,
            args.tokenizer,
            sp.dev_manifest,
            [float(x) for x in args.lm_grid.split(",")],
            [float(x) for x in args.ilm_grid.split(",")],
            args.tune_limit,
            workers,
        )
        weights = {
            m: (float(tuning[m]["lm_weight"] or 0.0), float(tuning[m]["ilm_weight"] or 0.0))
            for m in _MODES
        }
    else:
        # CLI override wins, else the configured value. The authoritative decode.yaml keeps
        # lm_weight=0.0 (the alpha=0 regression lock), so sweeps never mutate it.
        a = args.lm_weight if args.lm_weight is not None else cfg.decode.lm_weight
        b = args.ilm_weight if args.ilm_weight is not None else cfg.decode.ilm_weight
        weights = {m: (a, b) for m in _MODES}
        log(f"--- fixed weights (no dev sweep): alpha={a} beta={b}")
    for m in _MODES:
        log(f"    {m:<9} alpha={weights[m][0]} beta={weights[m][1]}")
    for stage in stages:
        if stage_uses_lm(stage) and max(weights[m][0] for m in _MODES) <= 0:
            # Otherwise this stage silently equals the pure-acoustic beam and the report misleads.
            log(
                f"WARNING: stage '{stage}' uses the LM but alpha=0 -> LM inactive; drop --no-tune "
                f"or pass --lm-weight > 0 to evaluate it"
            )

    plan = [(stage, mode) for stage in stages for mode in _MODES]

    def job(stage: str, mode: str, limit: int | None, timed: bool, tag: str) -> DecodePassJob:
        return DecodePassJob(
            kind="score",
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            stage=stage,
            mode=mode,
            manifest=sp.test_manifest,
            limit=limit,
            lm_weight=weights[mode][0],
            ilm_weight=weights[mode][1],
            measure_timing=timed,
            label=tag,
        )

    log(f"--- quality pass: {len(plan)} decode passes over n={n_test}, {workers} at a time ---")
    # Slowest first (beam_lm before greedy, streaming before offline): with fewer workers than
    # jobs, a long job started last is a long tail nothing overlaps.
    order = sorted(plan, key=lambda p: (-stages.index(p[0]), p[1] != "streaming"))
    quality = dict(
        zip(
            order,
            [
                cast(EvaluateCorpus_Response, r)
                for r in _dispatch(
                    [job(s, m, args.limit, False, f"{s}/{m}") for s, m in order], workers
                )
            ],
        )
    )

    rows = [quality[p] for p in plan]
    if probe_n > 0:
        log(
            f"--- timing pass: serial, n={probe_n} strided utts per configuration (RTF/latency are "
            f"only meaningful with the GPU to themselves) ---"
        )
        # workers=1 keeps this in THIS process, one job after another: whatever these numbers are,
        # they are one decoder's alone.
        probes = [
            cast(EvaluateCorpus_Response, r)
            for r in _dispatch(
                [job(s, m, probe_n, True, f"timing {s}/{m}") for s, m in plan], workers=1
            )
        ]
        rows = [
            replace(row, rtf=p.rtf, latency_s=p.latency_s, finalize_s=p.finalize_s)
            for row, p in zip(rows, probes)
        ]

    _print_table(rows)
    out = Path(args.report or cfg.eval.report_path.format(split=sp.name))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Which split produced these numbers, and where the weights came from, is part of the result --
    # a test-other table is not comparable to a test-clean one, and neither is comparable to a run
    # whose alpha was fixed by hand.
    out.write_text(
        json.dumps(
            {
                "split": sp.name,
                "manifest": sp.test_manifest,
                "tune_manifest": sp.dev_manifest if tuning else None,
                "checkpoint": args.checkpoint,
                "tokenizer": args.tokenizer,
                "beam_size": cfg.decode.beam_size,
                "chunk_size": cfg.decode.chunk_size,
                "length_bonus": cfg.decode.length_bonus,
                "weights": {
                    m: {"lm_weight": weights[m][0], "ilm_weight": weights[m][1]} for m in _MODES
                },
                "tuning": tuning,
                "timing": {
                    "measured": "serial, contention-free" if probe_n > 0 else "not measured",
                    "probe_utts": probe_n,
                    "quality_pass_workers": workers,
                },
                "wall_s": time.perf_counter() - t_start,
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    log(f"wrote {out}  (total {time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
