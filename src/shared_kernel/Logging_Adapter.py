# src/shared_kernel/Logging_Adapter.py — loguru sink setup (cross-cutting infra)
import sys

from loguru import logger

_CONFIGURED = False

# Timestamp + fixed-width level + message. Colorized for the terminal; the plain text
# survives a `tee` to a log file, which is how a multi-hour training run gets monitored.
_FORMAT = "<dim>{time:HH:mm:ss}</dim> │ <level>{level: <7}</level> │ <level>{message}</level>"


def configure_logging(level: str = "INFO"):
    """Idempotently point loguru at stderr with the project format. Returns the logger."""
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True)
        _CONFIGURED = True
    return logger
