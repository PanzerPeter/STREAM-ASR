# How much of the RNN-T lattice a batch stream actually spends on padding (CPU only, seconds).
#
# The full objective materialises [B, T, U+1, V], charged at the batch's *max* frames and *max*
# transcript length. Duration bucketing pins the first; nothing pins the second, so two utterances
# of equal length that differ 2x in speaking rate make every cell in the batch twice as expensive.
# `token_sort_window` re-sorts by transcript length inside a window of the duration sort to collapse
# that spread -- this script is how its value is chosen, and it reads the manifest only, so it costs
# no GPU and can be re-run whenever the budgets in config/training.yaml move.
import argparse
import statistics as st

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand


def _stats(sampler: FrameBucketSampler) -> tuple[int, float, float, float, float]:
    batches = sampler._batch_list()
    cells = [
        len(b) * max(sampler._frames[i] for i in b) * max(sampler._tokens[i] for i in b)
        for b in batches
    ]
    # "used" is what an unpadded lattice would cost: each utterance charged at its own frames and
    # its own transcript. The gap between the two is the padding the GPU pays for and learns from.
    used = sum(sampler._frames[i] * sampler._tokens[i] for b in batches for i in b)
    return (
        len(batches),
        st.mean(len(b) for b in batches),
        st.mean(cells),
        1.0 - used / sum(cells),
        float(sum(cells)),
    )


def main() -> None:
    base = TransducerTrainCommand()
    tr = get_config().training.transducer
    p = argparse.ArgumentParser(description="Lattice padding waste vs token_sort_window.")
    p.add_argument("--manifest", default=base.train_manifest)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 256, 1024, 4096])
    a = p.parse_args()

    print(
        f"  {a.manifest}\n  budgets: frames {tr.max_frames_per_batch:,} · "
        f"tokens {tr.max_tokens_per_batch:,} · lattice {tr.max_lattice_per_batch:.1e}\n"
    )
    header = f"  {'window':>7} {'batches':>9} {'mean B':>7} {'cells/batch':>12} {'waste':>7}"
    print(f"{header} {'epoch work':>11}")
    baseline = None
    for window in a.windows:
        sampler = FrameBucketSampler(
            a.manifest,
            tr.max_frames_per_batch,
            max_tokens_per_batch=tr.max_tokens_per_batch,
            max_lattice_per_batch=tr.max_lattice_per_batch,
            token_sort_window=window,
        )
        count, mean_b, mean_cells, waste, total = _stats(sampler)
        baseline = total if baseline is None else baseline
        # Epoch work, not cells/batch, is the number that matters: a narrower spread also packs
        # slightly fewer utterances per batch, so the per-batch saving overstates the real one.
        print(
            f"  {window:>7} {count:>9,} {mean_b:>7.2f} {mean_cells / 1e6:>11.2f}M "
            f"{waste * 100:>6.1f}% {total / baseline * 100:>10.1f}%"
        )


if __name__ == "__main__":
    main()
