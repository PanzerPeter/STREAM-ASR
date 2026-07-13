import os
import signal

from src.shared_kernel.SignalGuard import SignalGuard


def test_sigint_sets_flag_and_does_not_raise():
    with SignalGuard() as guard:
        assert guard.stop_requested is False
        os.kill(os.getpid(), signal.SIGINT)  # would raise KeyboardInterrupt without the guard
        assert guard.stop_requested is True


def test_restores_previous_handler_on_exit():
    before = signal.getsignal(signal.SIGINT)
    with SignalGuard():
        pass
    assert signal.getsignal(signal.SIGINT) is before
