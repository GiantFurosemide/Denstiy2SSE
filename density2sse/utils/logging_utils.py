"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure root logger with a simple stream handler."""
    log = logging.getLogger(name or "density2sse")
    if log.handlers:
        return log
    log.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(h)
    return log
