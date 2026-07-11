# src/slices/Evaluate/evaluate.py — corpus WER/CER/RTF/latency + ablation table (GPU; user-run).
# Loops the configured ablation stages x {offline, streaming}, building a decoder configured for
# each stage, and writes the full report to EvalConfig.report_path. Model/tokenizer loading mirrors
# src/slices/Decode/streaming_decode.py.
#
# LM evaluation: the LM only contributes at alpha (lm_weight) > 0. To evaluate it *honestly* the
# fusion weight must be tuned on a dev manifest and then frozen for the test run. `--tune DEV`
# does exactly that in one command: it sweeps `--lm-grid` on DEV, picks the alpha minimising the
# tune-stage WER, prints the sweep, then runs the full test table with that alpha. Without `--tune`,
# a fixed alpha comes from `--lm-weight` (or decode.lm_weight); alpha == 0 reproduces the pre-LM
# decoder exactly and the LM stages are flagged as inactive so the report never misleads.
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.Checkpoint_Adapter import load_checkpoint
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Evaluate.EvaluateCorpus_Command import EvaluateCorpus_Command
from src.slices.Evaluate.EvaluateCorpus_Handler import EvaluateCorpus_Handler
from src.slices.Evaluate.EvaluateCorpus_Response import EvaluateCorpus_Response

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
    use_rescore: bool
    fuse_lm_beam: bool
    fuse_lm_rescore: bool


# Cumulative ablation: greedy CTC -> prefix beam -> +attention rescore -> +LM in rescore -> +LM
# fusion in the first pass. lm_* stages require decode.lm_weight > 0 so the scorer is built.
_STAGES: dict[str, _StageFlags] = {
    "ctc_greedy": _StageFlags(1, False, False, False),
    "prefix_beam": _StageFlags(None, False, False, False),
    "attn_rescore": _StageFlags(None, True, False, False),
    "lm_rescore": _StageFlags(None, True, False, True),
    "lm_fusion": _StageFlags(None, True, True, True),
}


def _stage_uses_lm(stage: str) -> bool:
    f = _STAGES[stage]
    return f.fuse_lm_beam or f.fuse_lm_rescore


def _run_stage(
    model: HybridCtcAttention,
    tok: SentencePieceTokenizer,
    stage: str,
    lm_weight: float,
    manifest: str,
    mode: str,
    limit: int | None,
) -> EvaluateCorpus_Response:
    # Build a decoder configured for exactly this ablation stage + alpha, then score the corpus.
    # The decoder loads the LM only when a gate consumes it AND lm_weight > 0, so non-LM stages
    # (and alpha == 0) pay no LM cost.
    f = _STAGES[stage]
    decoder = StreamingDecoder_Handler(
        model,
        tok,
        beam_size=f.beam_size,
        use_rescore=f.use_rescore,
        fuse_lm_beam=f.fuse_lm_beam,
        fuse_lm_rescore=f.fuse_lm_rescore,
        lm_weight=lm_weight,
    )
    handler = EvaluateCorpus_Handler(decoder)
    # no_grad is thread-local: set it here so this holds whether _run_stage runs inline (tuning) or
    # inside a worker thread (the parallel table).
    with torch.no_grad():
        return handler.run(
            EvaluateCorpus_Command(
                manifest_path=manifest, mode=mode, ablation_stage=stage, limit=limit
            )
        )


def _tune_alpha(
    model: HybridCtcAttention,
    tok: SentencePieceTokenizer,
    dev_manifest: str,
    grid: list[float],
    stage: str,
    limit: int | None,
) -> float:
    # Sweep the fusion weight on a dev manifest and return the alpha minimising WER for the tuning
    # stage (offline mode: the cleanest, lowest-variance signal). Tuning on dev — never on the test
    # set that produces the headline number — keeps the reported WER an honest held-out result.
    print(f"--- tuning lm_weight on {dev_manifest} (stage={stage}, offline, n<= {limit}) ---")
    resps = _map_parallel(
        _run_stage,
        [(model, tok, stage, alpha, dev_manifest, "offline", limit) for alpha in grid],
    )
    best_alpha, best_wer = grid[0], float("inf")
    for alpha, resp in zip(grid, resps):
        marker = ""
        if resp.wer < best_wer:
            best_wer, best_alpha, marker = resp.wer, alpha, "  <- best"
        print(f"  lm_weight={alpha:<5} dev WER={resp.wer:.4f} CER={resp.cer:.4f}{marker}")
    print(f"--- selected lm_weight={best_alpha} (dev WER={best_wer:.4f}) ---")
    return best_alpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--checkpoint", default="data/checkpoints/stage_b_best.pt")
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
        help="cap dev utterances per alpha during --tune (a subset keeps the sweep affordable)",
    )
    ap.add_argument(
        "--tune-stage",
        default="lm_rescore",
        help="which LM ablation stage's dev WER the sweep minimises",
    )
    args = ap.parse_args()

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridCtcAttention()
    load_checkpoint(args.checkpoint, model)
    model = model.to(device).eval()
    tok = SentencePieceTokenizer(args.tokenizer)

    # _run_stage sets no_grad per task (thread-local), so no outer no_grad wrapper is needed.
    if args.tune is not None:
        grid = [float(x) for x in args.lm_grid.split(",")]
        lm_weight = _tune_alpha(model, tok, args.tune, grid, args.tune_stage, args.tune_limit)
    else:
        # CLI override wins, else the configured value. The authoritative decode.yaml keeps
        # lm_weight=0.0 (the alpha=0 regression lock), so sweeps never mutate it.
        lm_weight = args.lm_weight if args.lm_weight is not None else cfg.decode.lm_weight
    print(f"lm_weight (alpha) = {lm_weight}")

    report = []
    for stage in cfg.eval.ablation_stages:
        if _stage_uses_lm(stage) and lm_weight <= 0:
            # Otherwise this stage silently equals attn_rescore and the report misleads.
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
