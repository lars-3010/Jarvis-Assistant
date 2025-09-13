# ADR 0001 — Dependency Injection (DI) in MCP Server

Status: Proposed  
Date: 2025-09-11

## Context

- The MCP server currently supports two context implementations:
  - Traditional context constructing services directly (`src/jarvis/mcp/server.py:1`)
  - Container-aware context using DI (`src/jarvis/mcp/container_context.py:1`)
- The DI container centralizes service construction, configuration, and health, improving testability and consistency.
- Some duplication exists (e.g., tool registration paths), and adoption is partial.

## Decision

Adopt the container-aware MCP server context as the preferred path, with compatibility fallback to the traditional context via configuration (`JarvisSettings.use_dependency_injection`).

## Consequences

Positive:
- Consistent service lifecycle and configuration
- Easier testing/mocking of services
- Clearer integration points for extensions and analytics

Considerations:
- Ensure parity with traditional context (feature/health)
- Document migration and debugging steps

## Implementation Sketch

- Keep `use_dependency_injection=true` as the recommended default
- Validate health parity in startup logs
- Gradually integrate plugin registry for MCP tools to reduce manual registration duplication

## Alternatives Considered

- Continue manual construction only — simpler short term, but increases drift and complexity

