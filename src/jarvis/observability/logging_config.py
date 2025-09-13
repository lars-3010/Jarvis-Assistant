"""
Centralized logging configuration for Jarvis Assistant.

Provides a simple helper to initialize structured or plain logging
with optional file output. Keeps defaults quiet unless debug enabled.
"""

from __future__ import annotations

import logging
from logging import Handler, StreamHandler, FileHandler
from pathlib import Path
from typing import Optional


def configure_logging(level: str = "INFO", structured: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure root logger with optional structured formatting.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO")
        structured: If True, use JSON formatter when available
        log_file: Optional path to a log file
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicate logs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handlers: list[Handler] = [StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(FileHandler(str(log_file)))

    formatter = None
    if structured:
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore

            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
            )
        except Exception:
            # Fallback to plain formatter if json logger not available
            pass

    if formatter is None:
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    # Reduce noise from third-party libraries
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.debug("Logging configured: level=%s structured=%s file=%s", level, structured, str(log_file) if log_file else None)

