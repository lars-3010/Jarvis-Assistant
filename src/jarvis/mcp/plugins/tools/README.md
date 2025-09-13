MCP Tools
=========

Location
- All MCP tool implementations live in this directory.
- Do not place tool code under `src/jarvis/tools` — that folder is deprecated.

How tools are structured
- Each tool is a standalone module exposing an MCP action.
- Tools should:
  - Accept validated input (see `jarvis.mcp.schemas`),
  - Retrieve services via the DI container (`jarvis.core.container`),
  - Return structured JSON using `jarvis.mcp.structured` formatters/models.

References
- Container-aware context: `jarvis.mcp.container_context`
- Schemas: `jarvis.mcp.schemas` and `jarvis.mcp.structured.models`
- Formatters: `jarvis.mcp.structured.formatters`

Notes
- Keep imports targeting canonical services (e.g., `jarvis.services.vector`, `jarvis.services.analytics`).
- Prefer interfaces from `jarvis.core.interfaces` where possible.

