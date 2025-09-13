# Jarvis Assistant

Local‑first MCP server that lets AI systems (e.g., Claude Desktop) search, understand, and navigate your Obsidian vault — semantically and via relationships — with fast, structured JSON outputs designed for tool automation.

![High Level Architecture](resources/images/high-level-architecture.svg)

Project version: 0.2.0  
Docs snapshot: 2025-09-13

## What It Does

- Turns your vault into an AI‑friendly knowledge source with MCP tools.
- Answers queries using semantic similarity, exact text, and graph relations.
- Returns structured JSON so downstream automations can parse reliably.
- Works fully offline; optional Neo4j enables relationship queries.

## Goal & Use Cases

Jarvis Assistant’s goal is to make your personal knowledge base usable by AI agents as a reliable, automatable system of record — without sending your data to the cloud. It exposes your Obsidian vault via the Model Context Protocol (MCP) with predictable schemas so agents can search, reason, and act on your notes safely.

- Who it’s for: knowledge workers, researchers, and developers using Obsidian who want AI‑assisted retrieval, synthesis, and automation with privacy by default.
- Non‑goals: a generic chat bot or hosted SaaS; Jarvis is a local tool server focused on structured actions over your data.

Typical scenarios
- Research and Q&A: Ask conceptual questions; retrieve relevant passages and linked notes combining semantic + keyword + graph evidence.
- Domain exploration: Map knowledge domains, clusters, and bridges in a vault to discover gaps or connect ideas (Graph/Analytics tools).
- Note operations: Read specific notes with metadata, headings, and links so agents can cite, summarize, or refactor safely.
- Workflow automation: Use structured outputs (`schema_version`, `correlation_id`) in downstream scripts and automations.
- Health and observability: Inspect performance metrics and analytics cache freshness to keep agent behavior predictable.

## Key Capabilities

- **Semantic Search**: Meaning‑aware retrieval over your notes and headings
- **Graph Exploration**: Neighborhoods and relationships (Neo4j optional)
- **Hybrid Ranking**: Combine semantic + keyword evidence with analytics
- **Vault Ops**: Read notes, list vaults, surface health and performance
- **Structured Results**: Shared schemas, `schema_version`, `correlation_id`
- **Event‑Driven Analytics**: Cache invalidation on file changes for freshness

Why this approach
- Privacy: Local‑first by default; vectors in DuckDB; optional Neo4j for relationships.
- Reliability: All tools return structured JSON for deterministic parsing by agents.
- Extensibility: Plugin registry + schema manager make new tools easy to add safely.

## Prerequisites

- Python 3.11+
- UV package manager
- Obsidian vault
- Claude Desktop (for MCP integration)

## Quick Start

### A) Docker (recommended for demos)

```bash
# Build once
docker build -t jarvis-assistant:latest .

# Or use docker compose (mounts sample vault, data, logs by default)
docker compose up --build
```

Defaults (overridable via env):
- `JARVIS_VAULT_PATH=/vault`
- `JARVIS_DATABASE_PATH=/data/jarvis.duckdb`
- `JARVIS_VECTOR_DB_PATH=/data/jarvis-vector.duckdb`
- `JARVIS_LOG_FILE=/logs/mcp_server.log`

The compose file mounts the sample vault from `resources/sample_vault/` into `/vault`.

### B) Local (UV/Python)

```bash
git clone <repository-url>
cd jarvis-assistant

# Install deps (using uv)
uv sync

# Point to a vault (use the sample vault for demos)
export JARVIS_VAULT_PATH="$(pwd)/resources/sample_vault"
export JARVIS_DATABASE_PATH="$(pwd)/temporary/data/jarvis.duckdb"
export JARVIS_VECTOR_DB_PATH="$(pwd)/temporary/data/jarvis-vector.duckdb"
export JARVIS_LOG_FILE="$(pwd)/temporary/logs/mcp_server.log"

# Start the MCP server (stdio)
uv run jarvis-mcp-stdio
```

### Run Tests

```bash
# Install dependencies
uv sync

# Run the test suite (quiet)
uv run pytest -q
```

## Basic Usage Example

Once running, the MCP server provides these tools to Claude Desktop & other AI systems:

```bash
# Available MCP tools (JSON-only outputs)
- search-semantic         # Semantic search
- search-vault           # Keyword/filename/content search
- search-graph           # Relationship discovery around a note
- search-combined        # Hybrid (semantic + keyword) ranked
- search-graphrag        # GraphRAG MVP (semantic + neighborhoods + rerank)
- read-note              # Read specific files with metadata
- list-vaults            # Vault stats and model info
- get-health-status      # System health monitoring
- get-performance-metrics # Metrics grouped by category
- analytics-cache-status # Analytics cache/freshness
- analytics-invalidate-cache # Invalidate analytics cache
```

**Example interaction in Claude Desktop:**
> "Find notes about machine learning algorithms"
> 
> *Uses semantic search to find related concepts like neural networks, deep learning, AI, even if those exact words aren't in the query*

## Claude Desktop Integration

Add to your Claude Desktop configuration (`~/.claude.json`):

Option A — Local venv (stdio)

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/absolute/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio",
      "type": "stdio",
      "env": {
        "JARVIS_VAULT_PATH": "/absolute/path/to/your/vault",
        "JARVIS_DATABASE_PATH": "/absolute/path/to/jarvis.duckdb",
        "JARVIS_VECTOR_DB_PATH": "/absolute/path/to/jarvis-vector.duckdb",
        "JARVIS_LOG_FILE": "/absolute/path/to/mcp_server.log"
      }
    }
  }
}
```

Option B — Docker container (recommended for portability)

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "docker",
      "type": "stdio",
      "args": [
        "run", "--rm", "-i",
        "-v", "/absolute/path/to/your/vault:/vault:ro",
        "-v", "/absolute/path/to/data:/data",
        "-v", "/absolute/path/to/logs:/logs",
        "-e", "JARVIS_VAULT_PATH=/vault",
        "-e", "JARVIS_DATABASE_PATH=/data/jarvis.duckdb",
        "-e", "JARVIS_VECTOR_DB_PATH=/data/jarvis-vector.duckdb",
        "-e", "JARVIS_LOG_FILE=/logs/mcp_server.log",
        "jarvis-assistant:latest"
      ]
    }
  }
}
```

Notes
- Claude launches the container and connects over stdio; all data stays in mounted volumes.
- Ensure the image is built: `docker build -t jarvis-assistant:latest .` or `docker compose up --build`.
- On Windows, use absolute Windows paths; Docker Desktop will translate to the VM.

## Architecture Overview

```
Claude Desktop → MCP Server → Plugins (Registry) → Services → Databases/FS
                                   ├── Structured Formatters (JSON)
                                   ├── Vector Search (DuckDB)
                                   ├── Graph Search (Neo4j)
                                   ├── Observability (Metrics)
                                   └── Vault Access (FS)
```

- All tool outputs are structured JSON (schema_version=v1) and include a correlation_id.
- Tool discovery/execution is handled by a plugin registry (server is declarative).
- Analytics is event-driven (file changes → cache invalidation) and returns freshness fields.
- Canonical service implementations live under `src/jarvis/services/`.
- The former `monitoring/` has been renamed to `observability/` (metrics now; logging/tracing later).

Repository structure (condensed)

```
src/jarvis/
  core/            # DI, events, interfaces, registry
  services/        # canonical service implementations (analytics, graphrag, vector, graph, search, vault, health)
  mcp/             # MCP server and plugin system (tools in mcp/plugins/tools)
  database/        # duckdb (vector) and neo4j (graph) configuration/factories
  observability/   # metrics (and future logging/tracing)
  extensions/      # optional runtime extensions (e.g., AI/LLM)
  models/          # shared models
  utils/           # config, errors, helpers
  features/        # temporary shims; will be removed after deprecation
```

## Common Issues

- **Binary not found**: Run `uv sync` to create the virtual environment
- **Permission denied**: On Unix: `chmod +x .venv/bin/jarvis-mcp-stdio`
- **Vault not found**: Use absolute paths only
- **Server won't start**: Test with `uv run jarvis --help`
 - **Server won't start**: Test with `uv run jarvis-mcp-stdio --help`

## Documentation

- Architecture Map: `docs/architecture/architecture-map.md`
- arc42-style Docs: `docs/architecture/arc42.md`

Key Concepts
- Structured Responses: `src/jarvis/mcp/structured/`
- Plugin Registry: `src/jarvis/mcp/plugins/registry.py`
- Event Integration: `src/jarvis/core/event_integration.py`
- Analytics Engine: `src/jarvis/services/analytics/service.py`
- Search Services: `src/jarvis/services/search/`
- GraphRAG MVP: `src/jarvis/services/graphrag/`, tool: `src/jarvis/mcp/plugins/tools/search_graphrag.py`

AI Docs
- Project Overview (AI‑focused): `ai-docs/overview.md`
- MCP Tool Reference: `ai-docs/mcp-tools.md`

## Changelog

- v0.2.0 — 2025-09-13
  - Split heavy features into `src/jarvis/features/` with import shims under `src/jarvis/services/` (analytics, graphrag)
  - Integrated MCP `PluginRegistry` for list/execute; server is declarative
  - Defaulted to DI path; removed traditional context path
  - Stabilized structured responses; JSON format available across key tools
  - Added GraphRAG MVP tool (`search-graphrag`); continued performance/quality work
  - Maintained compatibility for analytics cache tools; event-driven invalidation planned
  - Began Pydantic v2 cleanup across models and extensions
  - Renamed `monitoring/` → `observability/`; `JarvisMetrics` moved accordingly

User Configuration
- Place overrides in `config/local.yaml`; base defaults in `config/base.yaml`.
- Environment overrides in `config/.env` (template: `config/.env.example`).
- Property extraction mapping (frontmatter keys, inline tag prefixes) is configurable.

Deployment (Docker)
- `Dockerfile` and `docker-compose.yaml` included.
- Volumes: `/vault` (read-only), `/data`, `/logs`.

## Contributing

1. Read the [Developer Guide](docs/05-development/developer-guide.md)
2. Follow the [Code Standards](docs/05-development/code-standards.md)
3. Add tests for new functionality
4. Update documentation for changes

## Security & CORS

- CORS origins (production): set explicit origins via env using JSON array syntax so Pydantic parses a list of strings correctly.
  - Shell: `JARVIS_CORS_ORIGINS='["https://your.app","https://admin.your.app"]' uv run jarvis-mcp-stdio`
  - config/.env: `JARVIS_CORS_ORIGINS='["https://your.app","https://admin.your.app"]'`
  - Docker Compose:
    - `environment: [ JARVIS_CORS_ORIGINS=["https://your.app","https://admin.your.app"] ]`

- Pre-commit hooks (uv):
  - Install: `uv tool install pre-commit && pre-commit install`
  - Generate secrets baseline: `uvx detect-secrets scan > .secrets.baseline && uvx detect-secrets audit .secrets.baseline && git add .secrets.baseline`
  - Run on all files: `pre-commit run --all-files`

- Dependency audit:
  - Quick: `uvx pip-audit`
  - Against exported lock: `uv export -o requirements.txt && uvx pip-audit -r requirements.txt`

## License

MIT License - see LICENSE file for details.

---

**Status**: Production ready with 8 working MCP tools  
**Next**: Enhanced performance and additional search capabilities  
**Support**: See [troubleshooting guide](docs/07-maintenance/troubleshooting.md) or open an issue
