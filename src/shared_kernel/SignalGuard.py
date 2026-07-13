# Cooperative interrupt handling for long training loops: catch SIGINT/SIGTERM, flip a flag, and let
# the loop reach its next safe point to checkpoint and exit — rather than tearing down mid-step and
# corrupting state. Pairs with the atomic Checkpoint_Adapter to make Ctrl-C safe (SP2).
import signal
import types
from typing import Iterable, Literal


class SignalGuard:
    def __init__(self, signals: Iterable[int] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self._signals = tuple(signals)
        self.stop_requested = False
        self._previous: dict[int, object] = {}

    def __enter__(self) -> "SignalGuard":
        self.stop_requested = False
        for s in self._signals:
            self._previous[s] = signal.getsignal(s)
            signal.signal(s, self._handle)
        return self

    def _handle(self, signum: int, frame: types.FrameType | None) -> None:
        self.stop_requested = True

    def __exit__(self, *exc: object) -> Literal[False]:
        for s, prev in self._previous.items():
            signal.signal(s, prev)  # type: ignore[arg-type]
        self._previous.clear()
        return False
