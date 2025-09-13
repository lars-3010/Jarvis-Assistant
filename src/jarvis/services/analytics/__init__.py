"""Analytics import shim.

This package re-exports analytics features from ``jarvis.features.analytics``
to maintain backwards compatibility with existing imports.
"""

from jarvis.features.analytics import (
    ActionableRecommendation,
    AnalyticsError,
    BridgeOpportunity,
    CacheStatus,
    DepthMetrics,
    DomainConnection,
    FolderHierarchy,
    KnowledgeDomain,
    OrganizationPattern,
    QualityGap,
    QualityScore,
    QualityTrend,
    VaultContext,
    VaultAnalyticsService,
)

__all__ = [
    "ActionableRecommendation",
    "AnalyticsError",
    "BridgeOpportunity",
    "CacheStatus",
    "DepthMetrics",
    "DomainConnection",
    "FolderHierarchy",
    "KnowledgeDomain",
    "OrganizationPattern",
    "QualityGap",
    "QualityScore",
    "QualityTrend",
    "VaultContext",
    "VaultAnalyticsService",
]
