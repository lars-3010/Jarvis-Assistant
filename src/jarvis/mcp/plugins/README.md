MCP Plugins
===========

Purpose
- Tool plugins exposed to MCP clients (e.g., Claude Desktop) via the MCP server.

Responsibilities
- Define tools with input schemas and execute them by leveraging services via DI.

Design Rules
- Keep tools thin; orchestrate services and return structured outputs.
- Register tools via the plugin discovery/registry in `mcp/plugins`.

Where to put tools
- All MCP tools live under `mcp/plugins/tools/`.
- Do not use `jarvis/tools` or `extensions/ai/tools` for MCP tools.
