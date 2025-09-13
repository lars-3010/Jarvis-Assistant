# MCP Plugin Schema Checklist

Use this checklist when adding or updating MCP plugins to ensure consistency,
machine-friendly defaults, and easy client integration.

- Schema templates: Use `jarvis.mcp.schemas` helpers
  - Search tools: `create_search_schema(SearchSchemaConfig)`
  - Graph tools: `create_graph_schema(...)`
  - Vault ops: `create_vault_schema(VaultSchemaConfig)`
  - Utility/Analytics: `create_utility_schema(UtilitySchemaConfig)` / `create_analytics_schema(...)`

- Output format: Always expose `format` and default to JSON
  - Use `supported_formats=["json", "markdown"]` (JSON first)
  - Read `format` with `arguments.get("format", "json").lower()`
  - Return JSON via structured formatters in `jarvis.mcp.structured` when `format=="json"`

- Structured payloads
  - Include `schema_version` and `correlation_id` in JSON payloads
  - Prefer formatter helpers: `*_to_json()` in `jarvis.mcp.structured`

- DI and services
  - Request services via container (`self.container.get(Interface)`) — no manual instantiation
  - Optional services (metrics/ranker) should be guarded with try/except

- Validation & limits
  - Enforce sensible bounds (e.g., `limit`, `depth`) in schema and at runtime
  - Validate required fields early and return helpful error messages

- Tests
  - Add/extend the registry smoke test to validate new schemas
  - Prefer unit tests that exercise tool argument parsing and formatter payloads

## Enforcement

- Lint check (manual): search for inline `inputSchema = { ... }` or direct JSON dicts in plugins.
  - Command: `rg -n "inputSchema=|input_schema\s*=\s*\{" src/jarvis/mcp/plugins/tools`
  - All tools should source schemas from `jarvis.mcp.schemas` helpers.
- Review: PRs adding tools must link to the schema helper used and the schema config.
- Test gate: registry smoke test should instantiate all tools and validate their `inputSchema` against JSON Schema draft (already covered by schema manager utilities).
