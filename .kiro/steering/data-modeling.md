# Data Modeling & Analytics Standards

## Data Modeling Philosophy
- **Structured Data as Lingua Franca**: All analytical outputs should be machine-readable JSON that enables complex AI reasoning rather than human-readable text summaries
- **Semantic Richness**: Data models should capture not just facts but relationships, confidence levels, and actionable insights
- **Progressive Enhancement**: Models should work with minimal data and gracefully enhance with additional context
- **Domain-Driven Design**: Models should reflect real-world knowledge management concepts and workflows

## Core Data Architecture Patterns

### Hierarchical Data Models
```python
# Base pattern for all analytical data
@dataclass
class AnalyticalResult:
    """Base class for all analytical results."""
    
    # Core metadata
    timestamp: float
    processing_time_ms: float
    confidence_score: float  # 0.0-1.0
    
    # Data freshness indicators
    cache_hit_rate: float
    data_freshness: str  # "current", "stale", "cached"
    
    # Error handling
    errors: List[AnalyticsError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### Quality Scoring Framework
```python
# Standardized quality assessment across all content types
@dataclass
class QualityScore:
    overall_score: float  # 0.0-1.0 normalized score
    level: QualityLevel   # 🌱🌿🌳🗺️ emoji indicators
    
    # Component scores (each 0.0-1.0)
    completeness: float    # Content depth and thoroughness
    structure: float      # Organization and formatting
    connections: float    # Links to other content
    freshness: float     # Recency and relevance
    
    # Actionable insights
    suggestions: List[str]
    confidence: float
```

### Connection and Relationship Modeling
```python
# Standard pattern for modeling relationships
@dataclass
class Connection:
    from_entity: str
    to_entity: str
    connection_type: str  # "hierarchical", "associative", "sequential"
    strength: float      # 0.0-1.0 connection strength
    bidirectional: bool
    
    # Context
    evidence: List[str]  # What indicates this connection
    confidence: float    # How certain we are about this connection
```

## Analytics Data Schemas

### Vault Context Schema
The primary analytical output that provides comprehensive vault understanding:

```python
@dataclass
class VaultContext:
    # Identity and basic stats
    vault_name: str
    total_notes: int
    total_size_bytes: int
    last_updated: float
    analysis_timestamp: float
    
    # Organizational intelligence
    organization_pattern: OrganizationPattern
    folder_structure: FolderHierarchy
    depth_metrics: DepthMetrics
    
    # Quality intelligence
    quality_distribution: Dict[str, int]  # {"🌱": 45, "🌿": 123, "🌳": 67, "🗺️": 12}
    average_quality_score: float
    quality_trends: List[QualityTrend]
    
    # Knowledge domain intelligence
    identified_domains: List[KnowledgeDomain]
    domain_connections: List[DomainConnection]
    isolated_notes: List[str]
    
    # Actionable insights
    recommendations: List[ActionableRecommendation]
    quality_gaps: List[QualityGap]
    bridge_opportunities: List[BridgeOpportunity]
    
    # System metadata
    processing_time_ms: float
    cache_hit_rate: float
    confidence_score: float
    analysis_complete: Dict[str, bool]
```

### Search Result Enhancement Schema
Enhanced search results with rich analytical metadata:

```python
@dataclass
class StructuredSearchResult:
    # Core result data
    path: str
    vault_name: str
    title: str
    preview: str
    
    # Scoring and relevance
    unified_score: float      # Combined relevance score
    confidence: float         # Confidence in result relevance
    relevance_factors: List[str]  # What made this relevant
    
    # Type-specific scores
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    graph_score: Optional[float] = None
    
    # Match information
    match_type: str  # "semantic", "keyword", "graph", "hybrid"
    match_positions: Optional[List[int]] = None
    match_context: Optional[str] = None
    
    # Content intelligence
    metadata: FileMetadata
    quality_indicators: QualityIndicators
    
    # Relationship context (for graph results)
    connections: Optional[List[Connection]] = None
    relationship_strength: Optional[float] = None
```

## Quality Assessment Standards

### Multi-Dimensional Quality Scoring
Quality assessment should evaluate multiple dimensions:

1. **Completeness** (0.0-1.0): Content depth and thoroughness
   - Word count relative to topic complexity
   - Presence of key sections (introduction, conclusion, examples)
   - Coverage of subtopics

2. **Structure** (0.0-1.0): Organization and formatting quality
   - Proper heading hierarchy
   - Use of lists and formatting
   - Logical flow and organization

3. **Connections** (0.0-1.0): Links to other content
   - Outbound link count and quality
   - Inbound link count (backlinks)
   - Bidirectional connections

4. **Freshness** (0.0-1.0): Recency and relevance
   - Last modification time
   - Content relevance to current context
   - Reference to recent information

### Quality Level Classification
```python
class QualityLevel(Enum):
    SEEDLING = "🌱"      # 0.0-0.25: Basic, incomplete content
    GROWING = "🌿"       # 0.25-0.50: Developing content with some structure
    MATURE = "🌳"        # 0.50-0.75: Well-developed, well-connected content
    COMPREHENSIVE = "🗺️" # 0.75-1.0: Comprehensive, authoritative content
```

### Quality Gap Identification
```python
@dataclass
class QualityGap:
    note_path: str
    current_quality: float
    potential_quality: float  # Estimated potential
    gap_type: str  # "completeness", "connections", "structure", "freshness"
    priority: str  # "high", "medium", "low"
    
    # Actionable guidance
    issues: List[str]
    suggestions: List[str]
    estimated_effort: str  # "5min", "30min", "2h", etc.
    
    # Context for improvement
    domain: Optional[str] = None
    related_notes: List[str] = field(default_factory=list)
```

## Knowledge Domain Modeling

### Domain Detection and Classification
```python
@dataclass
class KnowledgeDomain:
    name: str
    description: str
    note_count: int
    quality_distribution: Dict[str, int]  # emoji -> count
    average_quality: float
    
    # Semantic clustering
    keywords: List[str]
    semantic_clusters: List[SemanticCluster]
    representative_notes: List[str]  # Most characteristic notes
    
    # Structural patterns
    folder_paths: List[str]     # Common folder patterns
    file_patterns: List[str]    # Common naming patterns
    
    # Connection analysis
    internal_connections: int   # Links within domain
    external_connections: int   # Links to other domains
    isolation_score: float      # 0.0-1.0, higher = more isolated
    
    # Confidence and metadata
    confidence: float
    last_analyzed: float
```

### Cross-Domain Connection Analysis
```python
@dataclass
class DomainConnection:
    from_domain: str
    to_domain: str
    connection_strength: float  # 0.0-1.0
    connection_count: int
    bridge_notes: List[str]     # Notes that connect the domains
    connection_type: str        # "hierarchical", "associative", "sequential"
    
    # Improvement opportunities
    potential_bridges: List[tuple[str, str]]  # (note_a, note_b) pairs
    bridge_strategies: List[str]              # How to strengthen connection
```

## Performance and Caching Patterns

### Multi-Level Caching Strategy
```python
class AnalyticsCache:
    """Intelligent caching for analytics results."""
    
    # Cache Levels
    # L1: In-memory results (5 minutes TTL) - individual queries
    # L2: Computed analytics (1 hour TTL) - vault-level analysis
    # L3: Base statistics (24 hours TTL) - fundamental metrics
    
    # Cache Keys Pattern
    # {operation_type}:{vault_name}:{content_hash}
    # Examples:
    # - vault_context:default:a1b2c3d4
    # - quality_analysis:research:e5f6g7h8
    # - domain_map:work:i9j0k1l2
```

### Performance Monitoring
```python
@dataclass
class PerformanceMetrics:
    operation_name: str
    processing_time_ms: float
    memory_usage_mb: float
    cache_hit_rate: float
    
    # Resource utilization
    cpu_usage_percent: float
    disk_io_operations: int
    
    # Quality indicators
    result_count: int
    confidence_average: float
    error_count: int
```

## Error Handling and Resilience

### Structured Error Information
```python
@dataclass
class AnalyticsError:
    component: str          # Which analyzer failed
    error_type: str        # Classification of error
    message: str           # Human-readable description
    recovery_action: str   # Suggested recovery
    impact_level: str      # "low", "medium", "high"
    timestamp: float
    
    # Context for debugging
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
```

### Graceful Degradation Patterns
1. **Partial Analysis Mode**: If one analyzer fails, others continue
2. **Cached Fallback**: Return stale but valid cached results with freshness indicators
3. **Progressive Enhancement**: Start with basic metrics, add advanced analysis as available
4. **Error Context**: Include error details in response for debugging

## Data Validation and Consistency

### Schema Validation
- Use Pydantic models for all data structures
- Implement custom validators for domain-specific constraints
- Validate data consistency across related models
- Provide clear error messages for validation failures

### Data Integrity Checks
```python
def validate_vault_context(context: VaultContext) -> List[str]:
    """Validate vault context data integrity."""
    issues = []
    
    # Check score consistency
    if not (0.0 <= context.average_quality_score <= 1.0):
        issues.append("Average quality score out of range")
    
    # Check distribution consistency
    total_notes = sum(context.quality_distribution.values())
    if total_notes != context.total_notes:
        issues.append("Quality distribution doesn't match total notes")
    
    # Check confidence scores
    if not (0.0 <= context.confidence_score <= 1.0):
        issues.append("Confidence score out of range")
    
    return issues
```

## Testing Data Models

### Model Testing Patterns
```python
def test_quality_score_validation():
    """Test quality score model validation."""
    # Test valid score
    score = QualityScore(
        overall_score=0.75,
        level=QualityLevel.MATURE,
        completeness=0.8,
        structure=0.7,
        connections=0.6,
        freshness=0.9,
        suggestions=["Add more examples"],
        confidence=0.85
    )
    assert score.overall_score == 0.75
    
    # Test invalid score
    with pytest.raises(ValidationError):
        QualityScore(overall_score=1.5)  # Out of range
```

### Data Generation for Testing
```python
def create_test_vault_context(
    note_count: int = 100,
    quality_distribution: Optional[Dict[str, int]] = None
) -> VaultContext:
    """Create realistic test vault context."""
    if quality_distribution is None:
        quality_distribution = {"🌱": 20, "🌿": 40, "🌳": 30, "🗺️": 10}
    
    return VaultContext(
        vault_name="test_vault",
        total_notes=note_count,
        total_size_bytes=note_count * 1000,  # Approximate
        last_updated=time.time(),
        analysis_timestamp=time.time(),
        organization_pattern=create_test_organization_pattern(),
        folder_structure=create_test_folder_hierarchy(),
        depth_metrics=create_test_depth_metrics(),
        quality_distribution=quality_distribution,
        average_quality_score=calculate_average_from_distribution(quality_distribution),
        quality_trends=[],
        identified_domains=[],
        domain_connections=[],
        isolated_notes=[],
        processing_time_ms=150.0,
        cache_hit_rate=0.8,
        confidence_score=0.85,
        recommendations=[],
        quality_gaps=[],
        bridge_opportunities=[],
        cache_status=create_test_cache_status(),
        analysis_complete={
            "structure": True,
            "quality": True,
            "domains": True,
            "connections": True
        }
    )
```

## Configuration and Customization

### Analytics Configuration Schema
```yaml
analytics:
  enabled: true
  
  # Quality assessment configuration
  quality:
    scoring_algorithm: "comprehensive"  # "basic", "comprehensive"
    connection_weight: 0.3
    freshness_weight: 0.2
    completeness_weight: 0.3
    structure_weight: 0.2
    
    # Quality thresholds
    thresholds:
      seedling_max: 0.25
      growing_max: 0.50
      mature_max: 0.75
      comprehensive_min: 0.75
  
  # Domain analysis configuration
  domains:
    clustering_threshold: 0.7
    min_cluster_size: 3
    max_domains: 20
    semantic_similarity_threshold: 0.6
  
  # Performance configuration
  performance:
    max_processing_time_seconds: 15
    enable_parallel_processing: true
    sample_large_vaults: true
    sample_threshold: 5000
    
  # Caching configuration
  cache:
    enabled: true
    max_size_mb: 100
    l1_ttl_minutes: 5
    l2_ttl_minutes: 60
    l3_ttl_minutes: 1440  # 24 hours
```

## Migration and Versioning

### Schema Evolution Strategy
- Use semantic versioning for data model changes
- Maintain backward compatibility for at least 2 major versions
- Provide migration utilities for schema updates
- Document breaking changes and migration paths

### Data Migration Patterns
```python
def migrate_vault_context_v1_to_v2(old_context: Dict[str, Any]) -> VaultContext:
    """Migrate vault context from v1 to v2 schema."""
    # Handle schema changes
    if "organization_method" in old_context:
        # Convert old organization_method to new organization_pattern
        old_context["organization_pattern"] = OrganizationPattern(
            method=OrganizationMethod(old_context.pop("organization_method")),
            confidence=0.8,  # Default confidence
            indicators=[],
            folder_patterns=[],
            exceptions=[]
        )
    
    # Create new model with migrated data
    return VaultContext(**old_context)
```

## Documentation Standards

### Model Documentation Requirements
- Clear docstrings for all data classes and fields
- Examples of typical values and ranges
- Relationships between models
- Performance characteristics and limitations
- Validation rules and constraints

### API Documentation
- JSON schema definitions for all models
- Example requests and responses
- Error scenarios and handling
- Performance expectations
- Caching behavior documentation