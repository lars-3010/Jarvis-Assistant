"""
Database abstraction layer for Jarvis Assistant.

This module provides the factory pattern and adapters for multiple
database backends, enabling configuration-driven database selection.
"""

from .adapters import *
from .factory import DatabaseFactory
from .migration import DatabaseMigrator

__all__ = [
    "DatabaseFactory",
    "DatabaseMigrator",
]
