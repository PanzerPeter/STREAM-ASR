# BEST-RQ self-supervised encoder pretrain entry point (GPU; user-run).
import argparse
from dataclasses import replace

from src.slices.PretrainEncoder.BestRqPretrain_Command import BestRqPretrainCommand
from src.slices.PretrainEncoder.BestRqPretrainer_Handler import run_pretrain


def _parse_args() -> BestRqPretrainCommand:
    # Defaults come straight from the Command DTO (which itself reads config), so a bare invocation
    # is the production recipe. This stage used to have no CLI at all, which made starting a fresh
    # run mean editing the DTO -- the one thing a retrain always needs.
    base = BestRqPretrainCommand()
    p = argparse.ArgumentParser(description="BEST-RQ self-supervised Zipformer encoder pretrain.")
    p.add_argument("--train-manifest", default=base.train_manifest)
    p.add_argument("--dev-manifest", default=base.dev_manifest)
    # A cache is indexed by its manifest's row order and carries that manifest's fingerprint, so
    # these travel with the two flags above.
    p.add_argument("--cache-split", default=base.cache_split)
    p.add_argument("--dev-cache-split", default=base.dev_cache_split)
    p.add_argument("--cache-dir", default=base.cache_dir)
    p.add_argument("--total-steps", type=int, default=base.total_steps)
    p.add_argument("--log-dir", default=base.log_dir)
    p.add_argument("--ckpt-dir", default=base.ckpt_dir)
    p.add_argument("--device", default=base.device)
    p.add_argument("--num-workers", type=int, default=base.num_workers)
    p.add_argument(
        "--fresh", action="store_true", help="ignore any bestrq_last.pt and start from step 0"
    )
    a = p.parse_args()
    return replace(
        base,
        train_manifest=a.train_manifest,
        dev_manifest=a.dev_manifest,
        cache_split=a.cache_split,
        dev_cache_split=a.dev_cache_split,
        cache_dir=a.cache_dir,
        total_steps=a.total_steps,
        log_dir=a.log_dir,
        ckpt_dir=a.ckpt_dir,
        device=a.device,
        num_workers=a.num_workers,
        resume=not a.fresh,
    )


def main() -> None:
    out = run_pretrain(_parse_args())
    print(f"BEST-RQ pretrain finished. Encoder checkpoint: {out}")


if __name__ == "__main__":
    main()
