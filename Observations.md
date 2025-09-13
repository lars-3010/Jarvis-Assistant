Here’s a concise, critical pass over the current structure with concrete recommendations and a low-risk path to a cleaner layout.

  Core

  - State
      - Hosts DI (container.py), contracts (interfaces.py), events, service registry/discovery, task queue/scheduler.
  - Issues
      - Broad scope; “core” mixes DI, events, scheduling, and service discovery. Hard to scan and evolve.
  - Recommendations
      - Group by concern to improve navigability:
          - core/di (container)
          - core/contracts (interfaces)
          - core/events (events, integration)
          - core/services (service_registry, discovery)
          - core/scheduling (task_queue, task_scheduler)
      - Leave only stable, cross-cutting platform primitives in core. Avoid domain code here.

  Database

  - State
      - factory.py registers built-ins from services (DuckDB vector, Neo4j graph).
      - migration.py and connection_pool.py exist; the latter is substantial and async-capable, but not used by current services.
      - adapters/ is present but not active in the new simplified backend approach.
  - Issues
      - Pooling code appears unused; adds maintenance weight.
      - Split responsibilities between database and services/graph|vector can look redundant if not clearly documented.
  - Recommendations
      - Keep factory.py as the only entry in database/ for wiring backends. Move low-level backend code to services/* where it’s implemented (already how graph/vector do it).
      - Move connection_pool.py to database/_experimental/ or remove if not used; same for adapters/ (deprecate, then delete).
      - Move database-specific error types to database/errors.py (see “utils” below).

  Extensions

  - State
      - Plugin system for optional features (not MCP tools): AI/LLM routing, Graphrag extras, workflows, plus loader/registry/validation.
  - Issues
      - Overlaps with the idea of “plugins” in MCP. Two plugin systems can confuse contributors.
      - Feels close to “services.ai” (could be a service module) rather than a parallel subsystem unless the intent is clear (optional separately packaged features).
  - Recommendations
      - Clarify purpose in a short README (e.g., “runtime, optional features; not MCP-exposed tools”).
      - If these features are meant to be standard app services, consider folding into services/ai/* with clear interfaces in core/contracts. Keep “extensions” only for truly
  dynamically loaded, optional packages.
      - If “runtime extensions” is a requirement, standardize a single plugin substrate:
          - Extract a shared plugin kernel (registration/validation/discovery) into core/plugins and have both extensions and mcp/plugins build on it with adapters; or
          - Keep them separate but document the difference crisply (MCP tools vs runtime add-ons).

  MCP

  - State
      - Solid organization: plugins/, schemas/, structured/, server.py, container_context.py.
  - Issues
      - Duplicates plugin concepts with extensions (see above).
      - Prior logging divergence fixed; good.
  - Recommendations
      - If you keep two plugin systems, add a one-page doc/diagram clarifying boundaries (runtime vs client-exposed).
      - Consider moving plugins/registry.py’s generic pieces into a shared plugin kernel (optional; only if you want to reduce duplication).

  Models

  - State
      - models/document.py holds shared types (e.g., SearchResult).
      - Analytics data models live under services/analytics/models.py.
  - Issues
      - Mixed pattern: some domain models are shared in models/, others under service modules. This causes zig-zag imports (interfaces import models, services import
  interfaces).
  - Recommendations
      - Consolidate cross-service domain models under models/*:
          - Create models/analytics.py and move the analytics dataclasses there (or mirror them and deprecate the service-local module).
          - Provide an alias module services/analytics/models.py that re-exports from models.analytics for a release to avoid breakage.
      - Keep service-internal implementation details within services, but move pure data structures to models.

  Observability

  - State
      - Clean: metrics.py, logging_config.py, tracing.py.
  - Issues
      - None major. Clear separation; simple stubs for tracing.
  - Recommendations
      - Keep as-is. If it grows, subpackage: observability/logging, observability/metrics, observability/tracing for scale.

  Utils

  - State
      - config.py, errors.py, database_errors.py, helpers.py, user_config.py.
  - Issues
      - Catch-all; database_errors.py belongs with database; errors.py contains core-level error types used everywhere.
      - Helpers may hide responsibilities that belong to specific modules.
  - Recommendations
      - Split by concern:
          - Move DB-specific error types to database/errors.py.
          - Move core-wide error base classes to core/errors.py (and re-export from utils.errors for one release to avoid breakage).
          - Keep config.py where it is or move to core/config if you want to formalize it as platform-level. If moved, add a re-export shim.
          - Review helpers.py; either relocate into specific modules (e.g., path/logging helpers → observability or config) or split into utils/path.py, utils/io.py to avoid a
  “misc” bucket.

  Duplicate Responsibility Check

  - Core vs Utils: Errors and config straddle both. Move base errors to core; DB errors to database; leave only generic helpers in utils.
  - Extensions vs MCP: Overlapping plugin patterns. Either unify the plugin substrate or document the divergence clearly.
  - Database vs Services: Keep backend implementations in services; keep only the factory and backend registry in database.

  Low-Risk Refactor Plan

  - Step 1: Models
      - Create src/jarvis/models/analytics.py, move analytics dataclasses, and re-export from services/analytics/models.py to avoid breaking imports.
  - Step 2: Errors split
      - Create core/errors.py for shared error base types; create database/errors.py for DB-specific ones.
      - Re-export from utils/errors.py for backward compatibility.
  - Step 3: Database cleanup
      - Deprecate database/connection_pool.py and database/adapters/ with a note; remove in a future pass if unused.
  - Step 4: Core structure
      - Create subpackages: core/di, core/contracts, core/events, core/services, core/scheduling. Move files and add import shims in core/__init__.py for stability.
  - Step 5: Extensions vs MCP
      - Add a short doc explaining the two plugin systems.
      - Optionally, extract a minimal core/plugins kernel (interfaces, registry base) used by both, or defer and accept duplication with clear docs.

  If you want, I can draft the alias modules and shims (non-breaking), plus a short CONTRIBUTING-style doc page clarifying the boundaries.