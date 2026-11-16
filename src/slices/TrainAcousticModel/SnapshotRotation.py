# Checkpoint files beyond transducer_last.pt: rolling snapshots for averaging, and the kept
# branched-cooldown trunk points. Both are byte copies of the transducer_last.pt the same tick just
# wrote (same step, same RNG), so they resume cleanly; .tmp + os.replace, so never torn.
import glob
import os
import re
import shutil

_SNAPSHOT = re.compile(r"transducer_step(\d+)\.pt$")


def _copy(source: str, dest: str) -> None:
    tmp = dest + ".tmp"
    shutil.copyfile(source, tmp)
    os.replace(tmp, dest)


def _snapshots(ckpt_dir: str) -> list[tuple[int, str]]:
    found = glob.glob(os.path.join(ckpt_dir, "transducer_step*.pt"))
    return sorted((int(m.group(1)), p) for p in found if (m := _SNAPSHOT.search(p)))


def write_rolling_snapshot(ckpt_dir: str, source: str, step: int, keep_last_n: int) -> None:
    """transducer_step{N}.pt, pruned to the newest keep_last_n (0 disables) for averaging."""
    if keep_last_n <= 0:
        return
    _copy(source, os.path.join(ckpt_dir, f"transducer_step{step}.pt"))
    for _, path in _snapshots(ckpt_dir)[:-keep_last_n]:
        os.remove(path)


def write_branch_point(ckpt_dir: str, source: str, step: int) -> None:
    """transducer_branch{N}.pt: the trunk at a WSD decay start, never rotated (invariant 13)."""
    _copy(source, os.path.join(ckpt_dir, f"transducer_branch{step}.pt"))


def drop_snapshots_after(ckpt_dir: str, step: int, log) -> None:
    # A snapshot ahead of the resume point describes a future this run did not take; rotated by
    # step number it would outrank everything the run writes and be averaged into the decode model.
    for snap_step, path in _snapshots(ckpt_dir):
        if snap_step > step:
            os.remove(path)
            log.warning(f"removed {os.path.basename(path)}: ahead of the resumed step {step:,}.")
