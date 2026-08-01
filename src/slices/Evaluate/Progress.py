# Heartbeat logging for the corpus passes (infra for this slice only).
#
# Several decode passes run concurrently and each prints from its own thread, so a lock is what
# keeps one line from being torn in half by another, and the tag is what keeps the interleaved
# lines attributable. A corpus decode is otherwise a long silent grind: the first utterance proves
# the pipeline is live, then ~20 evenly spaced ticks carry elapsed/ETA plus whatever running metric
# the caller passes -- seeing WER drift on tick 3 beats discovering it an hour later.
import sys
import threading
import time

_PRINT_LOCK = threading.Lock()


def log(line: str) -> None:
    with _PRINT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


# Wide enough for the longest tag the driver builds ("timing greedy_transducer/streaming"), so the
# interleaved lines from concurrent passes stay column-aligned and scannable.
_TAG_W = 34


class Progress:
    def __init__(self, tag: str, total: int, ticks: int = 20) -> None:
        self.tag = tag
        self.total = total
        self.every = max(1, total // max(1, ticks))
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def tick(self, done: int, extra: str = "") -> None:
        if not (done == 1 or done == self.total or done % self.every == 0):
            return
        el = self.elapsed
        per = el / max(1, done)
        log(
            f"  [{self.tag:<{_TAG_W}}] {done:>5}/{self.total:<5}"
            f" {100 * done / max(1, self.total):>3.0f}%"
            f"  {_hms(el)} elapsed  {per:.2f}s/utt  ETA {_hms(per * (self.total - done))}"
            + (f"  {extra}" if extra else "")
        )

    def done(self, extra: str = "") -> None:
        log(
            f"  [{self.tag:<{_TAG_W}}] done   {self.total} utts in {_hms(self.elapsed)}"
            + (f"  {extra}" if extra else "")
        )
