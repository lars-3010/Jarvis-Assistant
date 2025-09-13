Current Status (Sep 2025)

- Canonical services: Analytics and GraphRAG now live under `src/jarvis/services/*`. The former `features/*` paths are shims that re‑export from services and emit DeprecationWarnings.
- Databases: The factory is streamlined to DuckDB (vector) and Neo4j (graph). Chroma/Pinecone adapters and related settings were removed. Docs updated accordingly.
- Observability: `monitoring/` was renamed to `observability/`. It currently hosts metrics; logging/tracing can be added here later.
- Docs: Top‑level and architecture docs now reflect the canonical stack and service locations.

High‑Level Findings

- Clear modular backbone: `core`, `services`, `database`, `utils`, `models`, and `mcp` are coherent and decoupled via interfaces and DI.
- Two plugin systems: `extensions` (runtime extensions) and `mcp/plugins` (MCP tool plugins). They’re distinct but can look overlapping without short docs.
- Tools path ambiguity: `src/jarvis/tools` exists with only a README, while actual tools live under `src/jarvis/mcp/plugins/tools`.
- Observability scope: Strong metrics implementation exists; room to include logging/tracing under the same umbrella when needed.

Overlaps and Ambiguities

- features vs services:
  - Canonical implementations reside in `services/*`.
  - `features/*` are temporary shims issuing DeprecationWarnings and re‑exporting from `services/*`.
- extensions vs mcp/plugins:
  - `extensions/` is for optional runtime add‑ons (e.g., LLM providers) that consume services.
  - `mcp/plugins` is specifically for MCP tool endpoints exposed to clients.
- tools naming:
  - `src/jarvis/tools` can be removed or replaced with a clear pointer to `mcp/plugins/tools`.

Recommended Restructure

- Services (canonical)
  - Keep all business logic in `services/*` (analytics, graphrag, vector, graph, search, vault, health).
  - Avoid circulars by depending on `core` interfaces and events; use `observability` for metrics.

- Database direction (simplified)
  - Continue with factory + registry, but only register DuckDB (vector) and Neo4j (graph).
  - Keep migration utilities; do not ship other adapters by default to reduce maintenance surface.

- Observability
  - Add `logging_config.py` (centralized logging setup) and a `tracing.py` stub when tracing becomes relevant. Keep metrics as is.

- MCP plugins
  - All tool implementations live under `mcp/plugins/tools`. Remove `src/jarvis/tools` or replace its README with an explicit pointer to MCP plugins.
  - Retain `mcp/tools` only for truly shared MCP tooling; otherwise consolidate into `mcp/plugins`.

- Documentation
  - Keep small README.md files in `services/`, `extensions/`, `mcp/plugins/`, and `observability/` clarifying purpose and dependency direction.

Suggested Migration Steps

- Phase 1 (complete)
  - Canonicalize Analytics and GraphRAG under `services/`.
  - Add shims and deprecation warnings in `features/`.
  - Update internal imports and top‑level docs.

- Phase 2 (complete)
  - Streamline database backends to DuckDB + Neo4j only; remove legacy adapters and settings; update docs.

- Phase 3 (complete)
  - Removed `src/jarvis/tools` and added a README under `mcp/plugins/tools`.
  - Removed deprecated `src/jarvis/features/*` shims.
  - Added observability stubs: `observability/logging_config.py`, `observability/tracing.py`.
  - Renamed analytics dataclass `AnalyticsError` → `AnalyticsIssue` and updated exports.
  - Added metrics instrumentation to `VectorSearcher.search` (records service operation, cache hits, and results gauge when IMetrics is available via DI).
  - Wired IMetrics into graph database operations (create/update and query) via container injection and recorded per-operation timings and failures.
  - Applied centralized logging configuration (`observability/logging_config.configure_logging`) in MCP entrypoint for consistent defaults.

Quick Wins

- Delete or repurpose `src/jarvis/tools` to eliminate ambiguity.
- Keep `features/*` shims with DeprecationWarnings for one grace release, then remove.
- Ensure GraphRAG and Analytics service outputs align with `mcp/structured/models.py` for strict schema validation in tools.

Longer‑Term Improvements

- Unify plugin metadata: Provide a small common metadata schema (name, version, capabilities, categories) shared by `extensions` and `mcp/plugins` to reduce duplication in registries, while keeping their interfaces separate.
- Observability expansion: Add centralized logging and a tracing stub; instrument vector/graph/analytics timing and error counters via IMetrics.
- Analytics model naming: Rename the dataclass `services.analytics.models.AnalyticsError` to `AnalyticsIssue` (or `AnalyticsErrorInfo`) to avoid confusion with runtime exceptions in `services.analytics.errors`.
- DI clarity: Consider removing direct fallback registrations in the container once factory creation is reliable across environments.

Optional (Follow‑Ups)

- Remove `src/jarvis/tools` and move its readme notes into `src/jarvis/mcp/plugins/tools/README.md`.
- Add `observability/logging_config.py` and a minimal `observability/tracing.py` stub.
- Rename the analytics dataclass to `AnalyticsIssue` and update exports/usages.
- Add IMetrics instrumentation to vector and graph services (timers, counters) and surface simple stats via an MCP tool.