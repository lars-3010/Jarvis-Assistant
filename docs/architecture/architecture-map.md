# Jarvis Assistant — Architecture Map (2025-09-11)

## High-Level View

Claude Desktop → MCP Server → Services → Databases / Filesystem

- MCP Server (`src/jarvis/mcp/server.py:1`)
  - Context: DI container (`src/jarvis/mcp/container_context.py:1`) is the default
  - Tool execution via Plugin Registry (list/execute)
  - Caching: `MCPToolCache`
  - Metrics: optional `JarvisMetrics`

- Services
  - Vector: `database`, `encoder`, `searcher`, `indexer`, `worker` under `src/jarvis/services/vector/`
  - Vault: `reader` under `src/jarvis/services/vault/reader.py:1`
  - Graph: `database` under `src/jarvis/services/graph/`
  - Analytics: orchestrator + analyzers under `src/jarvis/services/analytics/service.py:1`
  - Health/Observability: `src/jarvis/services/health.py`, `src/jarvis/observability/metrics.py`

- MCP Plugins
  - Tools under `src/jarvis/mcp/plugins/tools/` (combined, graph, list, health, etc.)
  - Plugin registry integrated (`src/jarvis/mcp/plugins/registry.py:1`) for listing and execution

- Extensions
  - Extension system under `src/jarvis/extensions/` (manager, loader, registry)
  - AI extension scaffold under `src/jarvis/extensions/ai/main.py:1` with LLM routing

- Core & Infra
  - Interfaces, container, events, task queue/scheduler under `src/jarvis/core/`
  - Settings via Pydantic under `src/jarvis/utils/config.py:1` (v2 cleanup in progress)

## Data Flows

- Semantic Search
  - MCP → `VectorSearcher.search()` → `ResultRanker` → formatted output

- Vault Read / Keyword Search
  - MCP → `VaultReader.search_vault()` or `read_file()` → formatted output

- Graph Search
  - MCP → `GraphDatabase.get_note_graph()` → grouping/formatting → output

- Analytics
  - MCP → `VaultAnalyticsService` (structure + quality + domains) → cache → formatted output (implementation in `jarvis.services.analytics`)

## Hotspots & Opportunities

- Structured Outputs
  - Today: mixed markdown and JSON-as-text patterns
  - Opportunity: add shared schemas/serializers and `format: "json"` across tools

- Event-Driven Analytics
  - Cache invalidation exists but not wired to file events → wire watcher/indexer events to analytics cache

- Tool Registration
  - Plugin registry exists; server still manually lists tools → integrate to reduce duplication

- DI Adoption
  - Dependency Injection is the default path; the traditional context has been removed. Use the container-aware context throughout.

## Key Files (by role)

- MCP Server: `src/jarvis/mcp/server.py:1`
- DI Context: `src/jarvis/mcp/container_context.py:1`
- Interfaces: `src/jarvis/core/interfaces.py:1`
- Analytics Service: `src/jarvis/services/analytics/service.py:1`
- Combined Search Tool: `src/jarvis/mcp/plugins/tools/search_combined.py:1`
- Graph Tool: `src/jarvis/mcp/plugins/tools/search_graph.py:1`
- Get Vault Context Tool: `src/jarvis/mcp/plugins/tools/get_vault_context.py:1`
- Assess Quality Tool: `src/jarvis/mcp/plugins/tools/assess_quality.py:1`
- Plugin Registry: `src/jarvis/mcp/plugins/registry.py:1`
- Extension Manager: `src/jarvis/extensions/manager.py:1`

## Next Steps

- Implement structured schemas/serializers and add `format: "json"` to all tools
- Wire events → analytics cache invalidation; add freshness/confidence fields
- Add GraphRAG MVP tool and pipeline
- Adopt plugin registry (integrated) as the single path for server tool registration and execution
