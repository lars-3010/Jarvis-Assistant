"""Analytics services (canonical)."""

from .models import (
    ActionableRecommendation,
    AnalyticsIssue,
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
    "AnalyticsIssue",
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
