# Data Modeling Standards

## Core Philosophy
- **Machine-Readable JSON**: Return structured data that enables AI reasoning, not just text summaries
- **Progressive Enhancement**: Models work with minimal data, enhance with more context
- **Confidence Scoring**: Include confidence levels (0.0-1.0) for all analytical results
- **Actionable Insights**: Provide specific suggestions, not just observations

## Core Data Patterns

### Base Analytical Result
All analytical outputs should include:
- `confidence_score` (0.0-1.0)
- `processing_time_ms`
- `data_freshness` ("current", "stale", "cached")
- `errors` and `warnings` lists

### Quality Scoring (🌱🌿🌳🗺️)
- **Overall Score**: 0.0-1.0 normalized
- **Components**: completeness, structure, connections, freshness
- **Levels**: 🌱 (0-0.25), 🌿 (0.25-0.5), 🌳 (0.5-0.75), 🗺️ (0.75-1.0)
- **Include**: actionable suggestions with confidence scores

### Connection Modeling
- `from_entity` → `to_entity` with `strength` (0.0-1.0)
- `connection_type`: "hierarchical", "associative", "sequential"
- Include `evidence` list and `confidence` score

## Key Data Schemas

### Vault Context (Primary Analytics Output)
- **Basic Stats**: vault_name, total_notes, total_size_bytes, last_updated
- **Quality Intelligence**: quality_distribution (🌱🌿🌳🗺️ counts), average_quality_score
- **Domain Intelligence**: identified_domains, domain_connections, isolated_notes
- **Actionable Insights**: recommendations, quality_gaps, bridge_opportunities
- **System Metadata**: processing_time_ms, cache_hit_rate, confidence_score

### Enhanced Search Results
- **Core Data**: path, vault_name, title, preview
- **Scoring**: unified_score, confidence, relevance_factors
- **Match Info**: match_type ("semantic", "keyword", "graph", "hybrid")
- **Quality**: metadata, quality_indicators
- **Relationships**: connections, relationship_strength (for graph results)

## Quality Assessment

### Quality Dimensions (Each 0.0-1.0)
1. **Completeness**: Content depth, key sections, subtopic coverage
2. **Structure**: Heading hierarchy, formatting, logical flow
3. **Connections**: Outbound/inbound links, bidirectional connections
4. **Freshness**: Modification time, content relevance

### Quality Gap Analysis
- Include `current_quality`, `potential_quality`, `gap_type`
- Provide `issues`, `suggestions`, `estimated_effort` ("5min", "30min", "2h")
- Add `priority` ("high", "medium", "low") and related notes context

## Domain Modeling
- **Domain Detection**: Include name, description, note_count, quality_distribution
- **Semantic Clustering**: keywords, representative_notes, folder_patterns
- **Connection Analysis**: internal_connections, external_connections, isolation_score
- **Cross-Domain**: connection_strength, bridge_notes, potential_bridges

## Caching & Performance
- **Multi-Level Cache**: L1 (5min), L2 (1hr), L3 (24hr)
- **Cache Keys**: `{operation_type}:{vault_name}:{content_hash}`
- **Performance Metrics**: processing_time_ms, memory_usage_mb, cache_hit_rate

## Error Handling
- **Structured Errors**: component, error_type, message, recovery_action, impact_level
- **Graceful Degradation**: Partial analysis mode, cached fallback, progressive enhancement
- **Error Context**: Include debugging context and stack traces when needed

## Validation & Testing
- **Schema Validation**: Use Pydantic models with custom validators
- **Data Integrity**: Validate score ranges (0.0-1.0), distribution consistency
- **Testing**: Test valid/invalid cases, create realistic test data factories