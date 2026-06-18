"""Lightweight logging setup.

A single :func:`get_logger` gives every module a consistently formatted logger
that writes to the console. Calling it repeatedly is safe — handlers are only
attached once.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "fractal", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Args:
        name:  Logger name (usually the module or pipeline stage).
        level: Logging threshold (e.g. ``logging.DEBUG``).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Attach a console handler only once, even across repeated calls.
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
