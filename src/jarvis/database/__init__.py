"""
Database abstraction layer for Jarvis Assistant.

This module provides the factory pattern for database backends.
Built-in support targets DuckDB (vector) and Neo4j (graph).
"""

from .factory import DatabaseFactory
from .migration import DatabaseMigrator

__all__ = [
    "DatabaseFactory",
    "DatabaseMigrator",
]
