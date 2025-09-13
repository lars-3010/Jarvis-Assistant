Adaptive context loader for Jarvis — load the right files fast.

Purpose: Quickly load the minimal, most relevant context based on a topic or task. Use with the read/grep tools to pull only what you need.

Usage:
- /context architecture
- /context mcp-tools
- /context schemas
- /context analytics
- /context events
- /context config
- /context tests

Topics → Files:
- architecture:
  - docs/architecture/arc42.md
  - docs/architecture/architecture-map.md
  - src/jarvis/mcp/server.py
  - src/jarvis/core/interfaces.py

- mcp-tools:
  - src/jarvis/mcp/plugins/tools/semantic_search.py
  - src/jarvis/mcp/plugins/tools/search_vault.py
  - src/jarvis/mcp/plugins/tools/search_graph.py
  - src/jarvis/mcp/plugins/tools/read_note.py
  - src/jarvis/mcp/plugins/tools/health_status.py
  - src/jarvis/mcp/plugins/tools/performance_metrics.py
  - src/jarvis/mcp/plugins/registry.py

- schemas:
  - src/jarvis/mcp/schemas/manager.py
  - src/jarvis/mcp/schemas/validator.py
  - src/jarvis/mcp/schemas/registry.py
  - src/jarvis/mcp/schemas/templates.py
  - src/jarvis/mcp/schemas/integration.py

- analytics:
  - src/jarvis/services/analytics/service.py
  - src/jarvis/services/analytics/cache.py
  - src/jarvis/services/analytics/analyzers/structure.py
  - src/jarvis/services/analytics/analyzers/quality.py
  - src/jarvis/services/analytics/analyzers/domain.py

- events:
  - src/jarvis/core/events.py
  - src/jarvis/core/event_integration.py

- config:
  - src/jarvis/utils/config.py
  - config/base.yaml
  - config/local.yaml
  - config/.env.example

- tests:
  - resources/tests/mcp/test_mcp_integration.py
  - resources/tests/integration/test_plugin_integration.py
  - resources/tests/unit/test_structured_formatters.py
  - resources/tests/unit/test_analytics_event_invalidation.py

Recommended Flow:
1) Identify topic → paste the relevant topic block
2) Use Read/Grep to pull only key sections (interfaces, execute(), schemas)
3) Summarize objectives, decisions, and exact code paths to modify
4) Implement surgically; update docs/tests alongside code

Notes:
- Prefer minimal reads; avoid loading entire large files unless necessary.
- For tool changes, always check schema/templates and structured formatters.
