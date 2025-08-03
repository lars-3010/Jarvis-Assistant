"""
Data models for vault analytics.

This module defines all data structures used by the analytics engine,
following the "Structured Data as Lingua Franca" principle.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum


class OrganizationMethod(Enum):
    """Supported vault organization methodologies."""
    PARA = "para"  # Projects, Areas, Resources, Archive
    JOHNNY_DECIMAL = "johnny_decimal"
    ZETTELKASTEN = "zettelkasten"
    TOPIC_BASED = "topic_based"
    CHRONOLOGICAL = "chronological"
    CUSTOM = "custom"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Quality levels with emoji indicators."""
    SEEDLING = "🌱"  # 0.0-0.25: Basic, incomplete content
    GROWING = "🌿"   # 0.25-0.50: Developing content with some structure
    MATURE = "🌳"    # 0.50-0.75: Well-developed, well-connected content
    COMPREHENSIVE = "🗺️"  # 0.75-1.0: Comprehensive, authoritative content


@dataclass
class OrganizationPattern:
    """Detected organization pattern in the vault."""
    method: OrganizationMethod
    confidence: float  # 0.0-1.0
    indicators: List[str]  # Evidence for this classification
    folder_patterns: List[str]  # Regex patterns found
    exceptions: List[str]  # Areas that don't fit the pattern


@dataclass
class FolderHierarchy:
    """Vault folder structure information."""
    max_depth: int
    average_depth: float
    total_folders: int
    root_folders: List[str]
    deepest_paths: List[str]  # Top 5 deepest paths
    empty_folders: List[str]  # Folders with no content


@dataclass
class DepthMetrics:
    """File and folder depth distribution metrics."""
    depth_distribution: Dict[int, int]  # depth -> count
    files_by_depth: Dict[int, List[str]]  # depth -> file paths
    complexity_score: float  # 0.0-1.0, higher = more complex structure
    organization_score: float  # 0.0-1.0, higher = better organized


@dataclass
class QualityTrend:
    """Quality trend over time."""
    timestamp: float
    average_quality: float
    quality_distribution: Dict[str, int]  # emoji -> count
    improvement_rate: float  # Change per day
    declining_notes: List[str]  # Notes with decreasing quality


@dataclass
class ConnectionMetrics:
    """Note connection analysis."""
    outbound_links: int
    inbound_links: int
    bidirectional_links: int
    broken_links: int
    connection_density: float  # 0.0-1.0
    hub_score: float  # How central this note is
    authority_score: float  # How referenced this note is


@dataclass
class QualityScore:
    """Comprehensive quality assessment for a note."""
    overall_score: float  # 0.0-1.0
    level: QualityLevel
    
    # Component scores (0.0-1.0 each)
    completeness: float    # Content depth and thoroughness
    structure: float      # Organization and formatting quality
    connections: float    # Links to other notes
    freshness: float     # Recency and relevance
    
    # Detailed metrics
    word_count: int
    link_count: int
    backlink_count: int
    last_modified: float
    headers_count: int
    list_items_count: int
    
    # Connection analysis
    connection_metrics: ConnectionMetrics
    
    # Improvement suggestions
    suggestions: List[str]
    confidence: float  # Confidence in the assessment
    
    # Context
    domain: Optional[str] = None  # Knowledge domain this note belongs to
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticCluster:
    """A cluster of semantically similar notes."""
    id: str
    centroid_note: str  # Most representative note
    notes: List[str]
    coherence_score: float  # 0.0-1.0, how similar notes are
    keywords: List[str]  # Key terms that define this cluster
    description: str


@dataclass
class KnowledgeDomain:
    """A distinct knowledge domain within the vault."""
    name: str
    description: str
    note_count: int
    quality_distribution: Dict[str, int]  # emoji -> count
    average_quality: float
    
    # Semantic information
    keywords: List[str]
    semantic_clusters: List[SemanticCluster]
    representative_notes: List[str]  # Top 5 most representative
    
    # Structure information
    folder_paths: List[str]  # Common folder patterns
    file_patterns: List[str]  # Common naming patterns
    
    # Connection information
    internal_connections: int  # Links within domain
    external_connections: int  # Links to other domains
    isolation_score: float  # 0.0-1.0, higher = more isolated
    
    # Metadata
    confidence: float  # Confidence in domain identification
    last_analyzed: float


@dataclass
class DomainConnection:
    """Connection between knowledge domains."""
    from_domain: str
    to_domain: str
    connection_strength: float  # 0.0-1.0
    connection_count: int
    bridge_notes: List[str]  # Notes that connect the domains
    connection_type: str  # "hierarchical", "associative", "sequential"


@dataclass
class QualityGap:
    """Identified quality improvement opportunity."""
    note_path: str
    current_quality: float
    potential_quality: float  # Estimated potential
    gap_type: str  # "completeness", "connections", "structure", "freshness"
    priority: str  # "high", "medium", "low"
    
    # Specific issues
    issues: List[str]
    suggestions: List[str]
    estimated_effort: str  # "5min", "30min", "2h", etc.
    
    # Context
    domain: Optional[str] = None
    related_notes: List[str] = field(default_factory=list)


@dataclass
class BridgeOpportunity:
    """Opportunity to connect related but unlinked domains."""
    domain_a: str
    domain_b: str
    similarity_score: float  # 0.0-1.0
    potential_connections: List[tuple[str, str]]  # (note_a, note_b) pairs
    rationale: str  # Why these domains should be connected
    priority: str  # "high", "medium", "low"
    
    # Implementation suggestions
    bridge_strategies: List[str]  # How to create connections
    seed_notes: List[str]  # Good starting points


@dataclass
class ActionableRecommendation:
    """Actionable recommendation for vault improvement."""
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    category: str  # "quality", "structure", "connections", "organization"
    
    # Implementation details
    action_items: List[str]
    estimated_time: str  # "5min", "1h", "1 day", etc.
    difficulty: str  # "easy", "medium", "hard"
    
    # Context
    affected_notes: List[str] = field(default_factory=list)
    affected_domains: List[str] = field(default_factory=list)
    
    # Metrics
    impact_score: float  # 0.0-1.0, expected improvement
    confidence: float  # 0.0-1.0, confidence in recommendation


@dataclass
class CacheStatus:
    """Analytics cache status information."""
    cache_hit_rate: float  # 0.0-1.0
    total_entries: int
    memory_usage_mb: float
    oldest_entry_age_minutes: float
    
    # Cache levels
    l1_entries: int  # In-memory (5 min TTL)
    l2_entries: int  # Computed analytics (1 hour TTL)
    l3_entries: int  # Base statistics (24 hour TTL)
    
    # Performance metrics
    average_lookup_time_ms: float
    cache_efficiency_score: float  # 0.0-1.0
    last_cleanup: float


@dataclass
class VaultContext:
    """Comprehensive vault context for AI analysis."""
    
    # Basic Statistics
    total_notes: int
    total_size_bytes: int
    last_updated: float
    analysis_timestamp: float
    vault_name: str
    
    # Organizational Structure
    organization_pattern: OrganizationPattern
    folder_structure: FolderHierarchy  
    depth_metrics: DepthMetrics
    
    # Quality Distribution  
    quality_distribution: Dict[str, int]  # {"🌱": 45, "🌿": 123, "🌳": 67, "🗺️": 12}
    average_quality_score: float
    quality_trends: List[QualityTrend]
    
    # Knowledge Domains
    identified_domains: List[KnowledgeDomain]
    domain_connections: List[DomainConnection]
    isolated_notes: List[str]  # Notes with few/no connections
    
    # Performance Metrics
    processing_time_ms: float
    cache_hit_rate: float
    confidence_score: float  # Overall confidence in analysis
    
    # Actionable Insights
    recommendations: List[ActionableRecommendation]
    quality_gaps: List[QualityGap] 
    bridge_opportunities: List[BridgeOpportunity]
    
    # System Information
    cache_status: CacheStatus
    errors: List['AnalyticsError'] = field(default_factory=list)
    
    # Analysis completeness indicators
    analysis_complete: Dict[str, bool] = field(default_factory=lambda: {
        "structure": False,
        "quality": False, 
        "domains": False,
        "connections": False
    })


@dataclass
class QualityAnalysis:
    """Detailed quality analysis results."""
    vault_name: str
    analysis_timestamp: float
    
    # Overall metrics
    average_quality: float
    quality_distribution: Dict[str, int]
    quality_trends: List[QualityTrend]
    
    # Detailed assessments
    note_scores: Dict[str, QualityScore]  # path -> quality score
    quality_gaps: List[QualityGap]
    improvement_priorities: List[str]  # Ordered by priority
    
    # Statistics
    processing_time_ms: float
    confidence_score: float
    cache_hit_rate: float


@dataclass
class DomainMap:
    """Knowledge domain mapping results."""
    vault_name: str
    analysis_timestamp: float
    
    # Domain information
    domains: List[KnowledgeDomain]
    domain_connections: List[DomainConnection]
    bridge_opportunities: List[BridgeOpportunity]
    
    # Clustering information
    semantic_clusters: List[SemanticCluster]
    isolated_notes: List[str]
    
    # Statistics
    processing_time_ms: float
    confidence_score: float
    cache_hit_rate: float


@dataclass 
class AnalyticsError:
    """Structured error information for analytics operations."""
    component: str          # Which analyzer failed
    error_type: str        # Classification of error
    message: str           # Human-readable description
    recovery_action: str   # Suggested recovery
    impact_level: str      # "low", "medium", "high"
    timestamp: float       # When the error occurred
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


# Type aliases for convenience
QualityDistribution = Dict[str, int]  # emoji -> count
DomainConnectionMap = Dict[str, List[DomainConnection]]  # domain -> connections
AnalyticsResult = Union[VaultContext, QualityAnalysis, DomainMap]