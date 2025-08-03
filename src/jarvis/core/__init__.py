"""
Core infrastructure for Jarvis Assistant.

This package contains the foundational components for dependency injection,
service management, and system architecture patterns.
"""

from .container import ServiceContainer
from .interfaces import (
    IVectorDatabase,
    IGraphDatabase,
    IVaultReader,
    IVectorEncoder,
    IVectorSearcher,
    IHealthChecker,
    IMetrics
)

__all__ = [
    "ServiceContainer",
    "IVectorDatabase",
    "IGraphDatabase", 
    "IVaultReader",
    "IVectorEncoder",
    "IVectorSearcher",
    "IHealthChecker",
    "IMetrics"
]