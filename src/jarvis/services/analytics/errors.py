"""Import shim for analytics errors (moved to features)."""

from jarvis.features.analytics.errors import (
    AnalyticsError,
    VaultNotFoundError,
    AnalysisTimeoutError,
    InsufficientDataError,
    CacheError,
    ServiceUnavailableError,
    ConfigurationError,
    ModelError,
    DataCorruptionError,
    ResourceExhaustionError,
)

__all__ = [
    "AnalyticsError",
    "VaultNotFoundError",
    "AnalysisTimeoutError",
    "InsufficientDataError",
    "CacheError",
    "ServiceUnavailableError",
    "ConfigurationError",
    "ModelError",
    "DataCorruptionError",
    "ResourceExhaustionError",
]

