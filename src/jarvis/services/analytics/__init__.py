"""
Analytics service for vault analysis and insights.

This module provides comprehensive analytical capabilities for vault structure,
content quality, and knowledge organization patterns.
"""

from .models import (
    VaultContext,
    QualityScore,
    KnowledgeDomain,
    OrganizationPattern,
    FolderHierarchy,
    DepthMetrics,
    QualityTrend,
    DomainConnection,
    ActionableRecommendation,
    QualityGap,
    BridgeOpportunity,
    CacheStatus,
    AnalyticsError,
)

__all__ = [
    "VaultContext",
    "QualityScore", 
    "KnowledgeDomain",
    "OrganizationPattern",
    "FolderHierarchy",
    "DepthMetrics",
    "QualityTrend",
    "DomainConnection",
    "ActionableRecommendation",
    "QualityGap",
    "BridgeOpportunity",
    "CacheStatus",
    "AnalyticsError",
]