# GraphRAG MVP — Requirements Document

## Introduction

Provide a minimal but practical GraphRAG pipeline that combines semantic retrieval with graph neighborhood expansion and structured, cited outputs — fully local-first and fast enough for interactive use.

## Requirements

### Requirement 1: Retrieval + Graph Expansion
**User Story:** As a researcher, I want answers grounded in both semantic similarity and graph relationships, so that I get context that is both relevant and well-connected.

Acceptance Criteria:
1. WHEN executing GraphRAG THEN the system SHALL retrieve semantic top-K and expand graph neighborhoods to depth N (bounded)
2. WHEN assembling contexts THEN the system SHALL prefer notes with strong graph connectivity (degree/path count) and recency if relevant
3. WHEN limiting scope THEN the system SHALL enforce caps on K, depth, and total tokens for performance

### Requirement 2: Structured Outputs with Citations
**User Story:** As an AI assistant, I want structured outputs with citations to sources and relationship paths, so that I can trace evidence and reason programmatically.

Acceptance Criteria:
1. WHEN returning results THEN the system SHALL provide JSON with: answer summary, sources (paths, vault, scores), graph paths (edges), and confidence
2. WHEN including analytics THEN the system SHALL report timings, selection metrics, and limits applied
3. IF insufficient signal THEN the system SHALL return a safe response indicating low confidence and surface alternative queries

### Requirement 3: Performance and Locality
**User Story:** As a user, I want GraphRAG to run quickly on my machine, so that I can use it interactively without cloud dependencies.

Acceptance Criteria:
1. WHEN running GraphRAG THEN the system SHALL complete within 12s for typical queries on mid-size vaults
2. WHEN caching THEN the system SHALL reuse previous retrieval/expansion results where possible
3. WHEN resource pressure is detected THEN the system SHALL degrade gracefully (smaller K/depth)

### Requirement 4: Integration and Safety
**User Story:** As a developer, I want GraphRAG to integrate cleanly with existing services, so that it’s maintainable and testable.

Acceptance Criteria:
1. WHEN integrating THEN the system SHALL reuse IVectorSearcher and IGraphDatabase; no new external dependencies required
2. WHEN exposing as a tool THEN the system SHALL provide a `search-graphrag` MCP tool with `mode`, `max_sources`, `include_citations`
3. WHEN errors occur THEN the system SHALL return partial results with diagnostics; failures SHALL not crash the server

