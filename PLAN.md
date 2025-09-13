# Jarvis Assistant — Master Plan, Analysis, and Refactoring Roadmap

Version: 2025-09-13 • Owner: Core Team • Status: Active

This single document consolidates and supersedes PLAN.md, analysis.md, and Refactoring.md. It combines strategy, architecture analysis, and the refactoring/migration plan into one living roadmap, updated to the current codebase state.

## Open Tasks

What remains open:
- Reduce remaining Pydantic v2 warnings (monitor during runtime/tests)
- Docs sweep for any remaining legacy references (continue incremental)
- Continue schema standardization as new tools are added; enforce via checklist/tests
- EventBus trio-compatibility: finalize cross-loop lifecycle and test stability in full suite

New findings (code audit — 2025‑09‑13):
- Schema standardization gaps:
  - Some tools (e.g., `search-combined`) define `inputSchema` inline. Migrate to `jarvis.mcp.schemas` templates and add a `format: "json|markdown"` switch for consistency.

---

## Current Status and Gaps

Done (high‑level):
- DI container path is default; traditional path removed
- MCP PluginRegistry integrated for list/execute
- Structured responses module in place; key tools support `format: "json"`
- Analytics and GraphRAG relocated to `features/` with shims to keep imports stable

Gaps:
- Analytics freshness: event‑driven invalidation not yet wired end‑to‑end
- GraphRAG: continue quality/perf iteration and docs
- Schema coverage: ensure all tools have clear JSON schemas and round‑trip validation
- Pydantic v2 deprecation warnings in a few areas
- DI container factory registration needs correction (see findings above)
- Logging config duplicated; standardize on a single path

---

## Architecture & Cleanup Analysis (merged)

High‑level layout (condensed):

```
src/jarvis/
  core/         # interfaces, DI, events, registry
  database/     # DB factories/adapters
  extensions/   # optional features (AI)
  mcp/          # server, plugins, schemas, structured outputs
  models/       # shared data models
  monitoring/   # metrics
  services/     # base services + import shims for moved features
  features/     # heavy/optional features (analytics, graphrag)
  utils/        # config, errors, helpers
```

Opportunities:
- Keep MCP server thin: plugin registry for discovery/execution (done)
- Structured outputs centralized in `mcp/structured` (ongoing expansion)
- Make heavy features optional (done via `features/` + shims)
- Normalize logging and reduce settings init noise
- Add small registry smoke test + schema round‑trip validation

---

## Roadmap & Workstreams (merged)

A) Structured Data Everywhere
- Define/maintain shared schemas and formatters
- Ensure all tools support `format: "json"` (markdown remains default)

B) Analytics + Events
- Subscribe analytics to file/vault events; invalidate caches intelligently
- Include freshness/confidence in all analytics responses
- Tools: `analytics-cache-status` and `analytics-invalidate-cache` (compat maintained)

C) GraphRAG MVP
- Pipeline: semantic top‑K → graph neighborhood expansion → rerank with graph features
- Tool: `search-graphrag` with `mode`, `max_sources`, `depth`, `include_content`
- Performance: strict limits, caching, logging

D) Context‑Aware Search (opt‑in)
- Session context store + contextual ranking/suggestions

E) Restructure & Refactor (Clarity & Velocity)
- Boundaries and naming consistency; extract shared structured code
- Docs refresh to reflect DI, plugin registry, features split
- Centralize logging configuration and remove ad‑hoc basicConfig usage

F) Config & Packaging
- Pydantic v2 idioms; secrets handling; containerization polish
- Add `register_factory` to container or relax signature when using factories

---

## Migration Plan (updated)

| Phase | Focus | Status |
|------|-------|--------|
| 0 | Structured scaffolding (`mcp/structured`) | Done |
| 1 | JSON coverage across tools | Mostly done; continue incremental |
| 2 | Event‑driven analytics invalidation | Planned |
| 3 | Plugin registry list/execute | Done |
| 4 | DI default path | Done |
| 5 | Boundary reshuffle (search svc, formatters) | In progress |
| 6 | GraphRAG MVP | In progress |
| 7 | Move heavy features to `features/` with shims | Done (analytics, graphrag) |
| 8 | Fix DI container factory registration | Planned |
| 9 | Unify logging configuration | Planned |
| 10 | Migrate extensions to Pydantic v2 patterns | Planned |

---

## Backlog (cleaned & prioritized)

Top priority
- AN‑EVT‑1: Verify event emitters and analytics subscription end‑to‑end; ensure cache invalidation triggers on file changes
- P2‑V2‑1: Reduce remaining Pydantic v2 warnings (ConfigDict/field serializers) across models
- DOC‑SWP‑1: Sweep legacy docs; clearly note `features/` split and import shims
- DI‑FIX‑1: Fix container factory registration for vector + graph DBs (add helper or adjust signature)
- LOG‑UNI‑1: Replace ad‑hoc logging setup with `configure_root_logging` across entry points

High
- SD‑COV‑2: Ensure all remaining tools support `format:"json"` with shared schemas
- GR‑MVP‑2: Iterate GraphRAG reranking features and result synthesis; add docs
- SCH‑STD‑2: Migrate plugins with inline schemas (e.g., `search-combined`) to `mcp.schemas` templates

Medium
- P2‑V2‑1: Reduce Pydantic v2 warnings (ConfigDict, json encoders replacements)
- LOG‑NORM‑1: Normalize logging config; reduce settings init verbosity
- DOC‑SWP‑1: Sweep legacy docs; add note about `features/` split and import shims
- EVT‑SAFE‑1: Add `ensure_event_bus_running()` no‑op guard for early publishes

Nice‑to‑have
- DEMO‑SV‑1: Sample Vault polish and demo scripts
 - TEST‑DI‑1: Add smoke tests for container factory registrations and error paths
 - TEST‑SCHEMA‑1: Round‑trip tests for schema validation and formatter outputs

Completed (recent)
- FEAT‑SPLIT‑1: Move analytics to `features/analytics` + import shims (done)
- FEAT‑SPLIT‑2: Move graphrag to `features/graphrag` + import shims (done)
- DI‑ANL‑1: Register `IVaultAnalyticsService` via container (when enabled) (done)
- MCP‑REG‑2: PluginRegistry integrated into MCP server list/execute (done)
- DI‑CORE‑2: DI default path; traditional context removed (done)
- DOC‑SWP‑0: Architecture docs updated to reflect `features/` + shims and registry integration (done)
- P2‑V2‑0: Initial Pydantic v2 cleanup: json_encoders replaced with field_serializer; .dict() → model_dump(); Config → ConfigDict in LLM models (done)

---

## Immediate Next Steps (1–2 days)

1) Analytics wiring
- Verify end‑to‑end event flow from watchers/indexer to analytics (cache invalidation)

2) Docs sweep
- Short note in `docs/` on new `features/` location and shims; update any remaining diagrams or pages

3) Pydantic v2 cleanup
- Continue replacing legacy Config and `.dict()` in remaining models

4) DI container bugfix
- Implement factory registration fix in `ServiceContainer` and adjust call sites; add smoke test

5) Logging unification
- Replace `basicConfig` usages with `configure_root_logging`; ensure no duplicate handlers

---

## Risks & Mitigations
- Latency in GraphRAG → strict limits, caching, streaming
- Schema drift → central schemas + round‑trip tests
- Deprecations → targeted v2 cleanup to reduce noise
- Confusion about locations → shims + docs sweep

---

## References
- MCP server, plugins, schemas: `src/jarvis/mcp/`
- Features (analytics, graphrag): `src/jarvis/features/`
- Service shims for compatibility: `src/jarvis/services/analytics`, `src/jarvis/services/graphrag`
- DI container: `src/jarvis/core/container.py`

---

End of document.
