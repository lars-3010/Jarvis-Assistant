Services
========

Purpose
- Canonical home for all service implementations that power Jarvis Assistant.

Responsibilities
- Business logic and orchestration across domains (search, vector, graph, analytics, graphrag, vault, health).
- Depend on `core/` for interfaces, events, DI; on `database/` for storage; on `observability/` for metrics.

Design Rules
- Keep domain logic here; do not place implementations in `core/`.
- Services may use models and utils; avoid circular dependencies.
- Prefer small, composable modules per domain (e.g., `vector/`, `graph/`, `search/`).

Notes
- Analytics and GraphRAG live here (migrated from `features/`).
- `features/` provides temporary import shims during the deprecation period.
