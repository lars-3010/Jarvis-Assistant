"""Analytics features package.

High-level analytics that consume vector, vault and graph services. This
namespace is intentionally separate from core services to preserve a lean core.

This package hosts the analytics implementation. Import shims under
``jarvis.services.analytics`` re-export these symbols to maintain backwards
compatibility.
"""

from .models import (
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
)

from .service import VaultAnalyticsService

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
