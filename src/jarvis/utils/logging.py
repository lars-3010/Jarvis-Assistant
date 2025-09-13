"""
Logging configuration for Jarvis Assistant.

This module provides centralized logging setup with structured logging support
and consistent formatting across all components.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

from pythonjsonlogger.json import JsonFormatter


def setup_logging(
    name: str | None = None,
    level: str | None = None,
    structured: bool = False,
    log_file: Path | None = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        name: Logger name (defaults to calling module)
        level: Logging level (INFO, DEBUG, etc.)
        structured: Enable JSON structured logging
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Determine logger name
    if name is None:
        name = __name__

    # Determine log level
    if level is None:
        level = "INFO"

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    if structured:
        formatter = JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_root_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Path | None = None
) -> None:
    """Configure root logging for the entire application.
    
    Args:
        level: Root logging level
        structured: Enable JSON structured logging
        log_file: Optional log file path
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "structured": {
                "()": "pythonjsonlogger.json.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "structured" if structured else "standard",
                "stream": "ext://sys.stderr"
            }
        },
        "root": {
            "level": level,
            "handlers": ["console"]
        },
        "loggers": {
            "jarvis": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        }
    }

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "structured" if structured else "standard",
            "filename": str(log_file)
        }
        config["root"]["handlers"].append("file")
        config["loggers"]["jarvis"]["handlers"].append("file")

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# Module-level convenience functions
def debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logging.getLogger("jarvis").debug(msg, *args, **kwargs)


def info(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logging.getLogger("jarvis").info(msg, *args, **kwargs)


def warning(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logging.getLogger("jarvis").warning(msg, *args, **kwargs)


def error(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    logging.getLogger("jarvis").error(msg, *args, **kwargs)


def critical(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    logging.getLogger("jarvis").critical(msg, *args, **kwargs)
