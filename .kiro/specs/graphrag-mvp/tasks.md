# GraphRAG MVP — Implementation Plan

## Phase 1: Retrieval and Expansion
- [ ] 1. Add retrieval wrapper
  - Create `GraphRAGRetriever` using `IVectorSearcher`
  - Configurable top-K, similarity threshold

- [ ] 2. Add neighborhood expansion
  - Create `GraphNeighborhoodFetcher` using `IGraphDatabase`
  - Configurable depth, caps (per-source and global)

## Phase 2: Scoring and Assembly
- [ ] 3. Implement feature extraction
  - Degree, edge count, path density, hop distance
- [ ] 4. Implement reranker
  - Linear combo of normalized semantic + graph features
- [ ] 5. Implement structured assembly
  - Build sources[], citations[], analytics

## Phase 3: MCP Tool
- [ ] 6. Add `search-graphrag` MCP tool
  - Arguments: query, mode, max_sources, include_citations, format
  - Text and JSON outputs; JSON includes full schema
  - Error handling with graceful fallbacks

## Phase 4: Caching and Performance
- [ ] 7. Add caching for neighborhoods and reranking
- [ ] 8. Add timing logs and simple metrics
- [ ] 9. Enforce limits by mode (quick/focused/comprehensive)

## Validation
- [ ] 10. Tests
  - Unit tests for fetcher, reranker, assembly
  - Integration test for tool with graph available and without (fallback)
- [ ] 11. Success metrics
  - < 12s E2E typical; citations correct; sources ranked sensibly

