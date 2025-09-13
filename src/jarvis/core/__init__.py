"""
Core infrastructure for Jarvis Assistant.

This package contains the foundational components for dependency injection,
service management, and system architecture patterns.
"""

from .container import ServiceContainer
from .interfaces import (
    IGraphDatabase,
    IHealthChecker,
    IMetrics,
    IVaultReader,
    IVectorDatabase,
    IVectorEncoder,
    IVectorSearcher,
)

__all__ = [
    "IGraphDatabase",
    "IHealthChecker",
    "IMetrics",
    "IVaultReader",
    "IVectorDatabase",
    "IVectorEncoder",
    "IVectorSearcher",
    "ServiceContainer"
]
