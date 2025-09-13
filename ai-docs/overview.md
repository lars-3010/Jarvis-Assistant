# Jarvis Assistant — AI Tooling Overview

Docs version: 0.2.0  
Last updated: 2025-09-13

## Summary

Jarvis Assistant exposes your Obsidian vault to AI systems through the Model Context Protocol (MCP). It provides deterministic, structured JSON outputs designed for automation, combines semantic, keyword, and graph search, and runs locally for privacy. Optional Neo4j enables relationship queries; DuckDB stores vectors by default.

## Goals

- Make personal knowledge reliably accessible to AI agents with strong guarantees (schemas, versions, metrics). 
- Preserve privacy with local‑first design and explicit, auditable data flows. 
- Enable downstream automations to parse and act on results deterministically.

## Non‑Goals

- General chat framework or hosted service. 
- Unstructured outputs that require brittle parsing.

## Capabilities

- Semantic retrieval over notes, headings, and content blocks. 
- Graph exploration of neighborhoods and relationships (optional Neo4j). 
- Hybrid ranking that merges semantic and keyword evidence with analytics. 
- Observability: health checks, performance metrics, analytics cache status.

## How AI Tools Should Use Jarvis

- Prefer structured JSON tool outputs when available; treat markdown as presentational only. 
- Use `schema_version` to branch logic; reject or fallback on unknown versions. 
- Capture `correlation_id` for traceability across chained tool calls. 
- Respect result limits (`limit`, `max_sources`); request more iteratively. 
- For graph features, start with shallow `depth` and expand as needed. 
- Use analytics endpoints to assess freshness or invalidate cache on major file changes.

## Output Conventions

- All tools include: `schema_version`, `correlation_id`, and an `analytics` block (if applicable). 
- Text content is returned with associated metadata (path, vault, headings, scores). 
- Errors are emitted as a single text item containing a structured JSON error object.

## Key Tools (see full reference)

Refer to `ai-docs/mcp-tools.md` for full parameter details.

- search-semantic — semantic retrieval over vault content. 
- search-vault — keyword/filename/content search. 
- search-graph — relationship discovery around a note. 
- search-combined — hybrid semantic + keyword ranking. 
- search-graphrag — combines semantic neighborhoods with reranking. 
- read-note — fetch a note and metadata. 
- list-vaults — vault stats and model info. 
- get-health-status — system health. 
- get-performance-metrics — tool/service timing and counters. 
- analytics-cache-status / analytics-invalidate-cache — analytics freshness controls.

## Environment & Configuration

- Required: `JARVIS_VAULT_PATH` points to the vault root. 
- Defaults (overridable): DuckDB paths for metadata and vectors; optional Neo4j if enabled. 
- See `config/base.yaml` (defaults), `config/local.yaml` (overrides), and `config/.env.example`.

## Versioning

- Project version: 0.2.0 (from `pyproject.toml`). 
- Docs snapshot: 2025-09-13. 
- Breakers: changes to schemas or tool names increment minor; response field additions are backward‑compatible.

## Pointers

- Architecture: `docs/architecture/arc42.md`, `docs/architecture/architecture-map.md`. 
- Structured models: `src/jarvis/mcp/structured/`. 
- Plugin registry: `src/jarvis/mcp/plugins/registry.py`. 
- Schema manager: `src/jarvis/mcp/schemas/`. 
- Analytics (implementation): `src/jarvis/features/analytics/` (re-exported via `src/jarvis/services/analytics/`). 
- GraphRAG MVP (implementation): `src/jarvis/features/graphrag/` (re-exported via `src/jarvis/services/graphrag/`).

## Changelog

- See `PLAN.md` for roadmap themes and upcoming changes. Align AI behaviors with `schema_version` and prefer capability detection over assumptions.
