# ADR 0002 — Structured Responses Module

Status: Proposed  
Date: 2025-09-11

## Context

- MCP tools primarily return markdown; some tools can emit JSON as text
- Lack of shared schemas leads to duplication and fragile parsing
- Need to support both humans (markdown) and AIs (JSON) cleanly

## Decision

Introduce a shared structured responses module at `src/jarvis/mcp/structured/` that provides:
- Pydantic models for common response shapes (semantic, keyword, graph, combined; vault stats; health)
- Serializer helpers to convert internal objects to schema dicts
- A `format` parameter across tools; default markdown for backward compatibility
- Default JSON transport as `TextContent` payload for MCP compatibility (revisit `JsonContent` later)

## Consequences

Positive:
- Consistent, testable structured outputs
- Drop-in upgrade path for existing tools via shared helpers

Considerations:
- Minor overhead; ensure performance targets via caching/limits

## Implementation Sketch

- Create `src/jarvis/mcp/structured/models.py`, `formatters.py`
- Add unit tests for schema validation and serialization
- Update tools incrementally to accept `format: "json"` and serialize via helpers

## Alternatives Considered

- Ad-hoc JSON per tool — faster initially, but leads to drift and brittle clients

