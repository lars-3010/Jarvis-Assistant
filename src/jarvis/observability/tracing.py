"""
Lightweight tracing stubs for Jarvis Assistant.

This module provides no-op tracing utilities that can be expanded later
to integrate with real tracing backends. For now, it logs spans at DEBUG.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def span(name: str) -> Generator[None, None, None]:
    """Context manager to trace a logical operation span."""
    start = time.time()
    logger.debug("[trace] span.start name=%s", name)
    try:
        yield
    finally:
        dur = (time.time() - start) * 1000.0
        logger.debug("[trace] span.end name=%s duration_ms=%.2f", name, dur)


def traced(func):
    """Decorator to trace a function call as a span."""
    def wrapper(*args, **kwargs):
        with span(func.__name__):
            return func(*args, **kwargs)
    return wrapper

