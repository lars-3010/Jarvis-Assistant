Extensions
==========

Purpose
- Optional runtime extensions that integrate with the core system via DI (e.g., AI/LLM providers).

Responsibilities
- Provide additional capabilities without being required for core operation.
- Register extension metadata and expose optional MCP tools if applicable.

Design Rules
- Extensions should depend on `services/` and `core/`, not the other way around.
- Keep provider-specific code here; keep generic service logic under `services/`.

Notes
- MCP tool plugins are not here; they live under `mcp/plugins/tools`.
