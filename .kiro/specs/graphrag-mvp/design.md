# GraphRAG MVP — Design Document

## Overview

Pipeline:
1. Semantic Retrieval: top-K candidates using `IVectorSearcher`
2. Graph Expansion: for each candidate, fetch neighborhood to bounded depth with `IGraphDatabase`
3. Feature Extraction: compute graph features (degree, edge count, path overlap, hop distance)
4. Reranking: combine semantic score + graph features (weighted)
5. Assembly: select top-N sources, build structured output with citations and relationship paths

## Components

### Retrieval
- Service: `IVectorSearcher`
- Config: K (default 10), similarity threshold (optional)

### Graph Expansion
- Service: `IGraphDatabase`
- Config: depth (default 1–2), max nodes/edges per source, global caps

### Ranking
- Heuristics: normalized semantic score + connectivity (degree, edges) + path density
- Simple weighted linear combo for MVP; future: learnable weights

### Output Schema
- `answer`: generated or synthesized summary (optional for MVP; can be a structured set of sources + rationale)
- `sources[]`: { path, vault, semantic_score, graph_score, unified_score }
- `citations[]`: { source_path, relationships: [{source,target,type}] }
- `analytics`: { execution_time_ms, limits_applied, counts, confidence }

## Tool Interface

Tool: `search-graphrag`
Arguments:
- `query: string`
- `mode: "quick"|"focused"|"comprehensive"` (sets K/depth/limits)
- `max_sources: int` (default 10)
- `include_citations: boolean` (default true)
- `format: "markdown"|"json"` (default markdown)

## Performance
- Enforce tight caps; stream assembly when possible
- Cache retrieval and neighborhoods keyed by (path, depth)

## Risks
- Graph availability: fallback to semantic-only with explanation
- Latency: mitigate with strict limits and caching

