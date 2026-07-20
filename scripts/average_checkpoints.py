# scripts/average_checkpoints.py — mean the tail of training's transducer_step{N}.pt snapshots into
# one decode checkpoint (standard ASR checkpoint averaging). Point config/decode.yaml or
# config/eval.yaml at the output. Snapshots come from TransducerTrainer's keep_last_n retention.
import argparse
import glob
import os
import re

from src.shared_kernel.Checkpoint_Adapter import average_checkpoints


def _newest_snapshots(ckpt_dir: str, n: int) -> list[str]:
    paths = glob.glob(os.path.join(ckpt_dir, "transducer_step*.pt"))
    numbered = sorted(
        ((int(m.group(1)), p) for p in paths if (m := re.search(r"step(\d+)\.pt$", p))),
        key=lambda x: x[0],
    )
    if not numbered:
        raise SystemExit(f"no transducer_step*.pt snapshots in {ckpt_dir}")
    return [p for _, p in numbered[-n:]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default="data/checkpoints")
    ap.add_argument("--last-n", type=int, default=5, help="average the newest N snapshots")
    ap.add_argument(
        "--inputs", nargs="*", help="explicit checkpoint paths (overrides --ckpt-dir/--last-n)"
    )
    ap.add_argument("--out", default="data/checkpoints/transducer_avg.pt")
    args = ap.parse_args()

    paths = args.inputs if args.inputs else _newest_snapshots(args.ckpt_dir, args.last_n)
    average_checkpoints(paths, args.out)
    print(f"averaged {len(paths)} checkpoints -> {args.out}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
