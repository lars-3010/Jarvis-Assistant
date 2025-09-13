# Jarvis Assistant — arc42 Architecture Documentation

Version: 1.0 • Date: 2025-09-11

## 1. Introduction and Goals

- Local-first MCP server integrating semantic search, graph exploration, and analytics.
- Goals: structured JSON outputs, plugin-based extensibility, privacy, performance, testability.

## 2. Constraints

- JSON-only outputs (schema_version=v1) with correlation_id.
- Must run fully local; optional Dockerized deployment.
- Pluggable services via DI/container and plugin registry.

## 3. System Context

```mermaid
flowchart LR
  Claude[Claude Desktop / MCP Client] -->|MCP| Server[MCP Server]
  Server --> Registry[Plugin Registry]
  Registry --> Plugins[Tool Plugins]
  Plugins --> Structured[Structured Formatters]
  Plugins --> Services
  Services --> VectorDB[(DuckDB)]
  Services --> GraphDB[(Neo4j)]
  Services --> VaultFS[(Filesystem)]
  EventBus[(Event Bus)] --> Services
```

## 4. Solution Strategy

- SOLID:
  - SRP: Plugins orchestrate; services compute; formatters present.
  - OCP: Add new tools/services without modifying existing components (registry + DI).
  - LSP/ISP: Narrow interfaces (`IVectorSearcher`, `IGraphDatabase`, `IVaultReader`, `IVaultAnalyticsService`).
  - DIP: Plugins depend on interfaces resolved by DI.
- Structured responses: centralized schemas and helpers; versioned payloads.
- Event-driven analytics: file changes invalidate cache, analytics return freshness.

| Principle | Mapping |
|---|---|
| SRP | `plugins`, `services`, `structured` separated roles |
| OCP | Plugin registry + DI for extensions |
| LSP | Replaceable service implementations via interfaces |
| ISP | Multiple small ports (searcher, graph, vault, analytics) |
| DIP | Composition root injects all ports into plugins |

## 5. Building Block View

```mermaid
classDiagram
  class MCPServer {+list_tools() +call_tool()}
  class PluginRegistry {+get_tool_definitions() +execute_tool()}
  class MCPToolPlugin <<interface>>
  class StructuredFormatters {+to_json()}
  class IVectorSearcher <<interface>>
  class IGraphDatabase <<interface>>
  class IVaultReader <<interface>>
  class IVaultAnalyticsService <<interface>>
  class VaultAnalyticsService
  class ResultRanker
  class GraphRAGRetriever
  class GraphNeighborhoodFetcher
  class GraphRAGReranker

  MCPServer --> PluginRegistry
  PluginRegistry --> MCPToolPlugin
  MCPToolPlugin --> StructuredFormatters
  MCPToolPlugin ..> IVectorSearcher
  MCPToolPlugin ..> IVaultReader
  MCPToolPlugin ..> IGraphDatabase
  MCPToolPlugin ..> IVaultAnalyticsService
  IVaultAnalyticsService <|.. VaultAnalyticsService
  GraphRAGRetriever ..> IVectorSearcher
  GraphNeighborhoodFetcher ..> IGraphDatabase
```

Note: Heavy features (analytics, GraphRAG) reside under `jarvis.features.*` and are re-exported via import shims at `jarvis.services.*` to keep import paths stable. The diagrams show service classes generically; consult `src/jarvis/features/` for implementations.

## 6. Runtime View

### 6.1 Tool Invocation

```mermaid
sequenceDiagram
  participant Client
  participant Server
  participant Registry
  participant Plugin
  participant Services
  Client->>Server: list_tools
  Server->>Registry: get_tool_definitions
  Registry-->>Server: Tool[]
  Server-->>Client: Tool[] (with correlation_id in logs)
  Client->>Server: call_tool(name,args)
  Server->>Registry: execute_tool(name,args)
  Registry->>Plugin: execute(args)
  Plugin->>Services: query/index/graph
  Services-->>Plugin: results
  Plugin-->>Registry: TextContent(JSON)
  Registry-->>Server: TextContent(JSON)
  Server-->>Client: TextContent(JSON)
```

### 6.2 Analytics Invalidation

```mermaid
sequenceDiagram
  participant Worker as VectorWorker
  participant EventBus
  participant Analytics
  Worker->>EventBus: DOCUMENT_UPDATED/DELETED
  EventBus->>Analytics: notify(event)
  Analytics->>Analytics: cache.invalidate(vault)
  Client->>Analytics: get_vault_context
  Analytics-->>Client: fresh JSON (cache_hit=false, content_hash)
```

### 6.3 GraphRAG MVP

```mermaid
sequenceDiagram
  participant Tool as search-graphrag
  participant Retriever
  participant Fetcher
  participant Reranker
  Tool->>Retriever: retrieve(query, topK)
  Retriever-->>Tool: semantic[]
  loop Top Sources
    Tool->>Fetcher: fetch(center, depth)
    Fetcher-->>Tool: graph
  end
  Tool->>Reranker: rerank(semantic, graphs)
  Reranker-->>Tool: ranked[]
  Tool-->>Client: JSON (sources, graphs, schema_version, correlation_id)
```

## 7. Deployment View

```mermaid
flowchart LR
  subgraph Container
    App[Jarvis MCP Server]
    Vol1[/vault/]
    Vol2[/data/]
    Vol3[/logs/]
  end
  Client[Claude Desktop] -->|MCP| App
  Vol1 --- App
  Vol2 --- App
  Vol3 --- App
```

Docker:
- Dockerfile and docker-compose.yaml provided; mount sample vault and data/logs.

## 8. Cross-cutting Concepts

- Structured JSON Responses: centralized models/formatters; schema_version & correlation_id.
- Config & Secrets: YAML config (base/local) with overrides; pluggable secrets provider (future).
- Observability: structured logs, correlation IDs, metrics categorized by prefixes.

## 9. Quality Requirements

- Performance targets: <2s cached analytics, <15s fresh; <12s GraphRAG MVP.
- Stability: event-driven invalidation; resilient error handling; partial results with diagnostics.

## 10. Risks & Technical Debt

- Graph DB availability: tools fallback with explanation; consider abstractions for alternate graph stores.
- Learning curve: provide sample vault, dockerized demos, JSON schemas.
