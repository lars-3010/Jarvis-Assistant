# Jarvis Assistant

MCP server providing semantic search and graph analysis tools for AI systems to intelligently discover knowledge in Obsidian vaults.

![High Level Architecture](resources/images/high-level-architecture.svg)

## Key Features

- **Semantic Search**: Find related content through meaning, not just keywords
- **Graph Relationships**: Discover connections between notes and concepts
- **Claude Desktop Integration**: Direct MCP protocol integration with AI tools
- **Local-First**: All processing happens locally for privacy and control
- **Production Ready**: 8 working MCP tools with robust error handling

## Prerequisites

- Python 3.11+
- UV package manager
- Obsidian vault
- Claude Desktop (for MCP integration)

## Quick Start

### 1. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies
```bash
git clone <repository-url>
cd jarvis-assistant
uv sync
```

### 3. Index Your Vault
```bash
uv run jarvis index --vault /path/to/your/obsidian/vault
```

### 4. Start MCP Server
```bash
uv run jarvis mcp --vault /path/to/vault --watch
```

## Basic Usage Example

Once running, the MCP server provides these tools to Claude Desktop & other AI systems:

```bash
# Available MCP tools (8 production-ready tools):
- search-semantic     # Find related content: "productivity techniques"
- search-vault       # Keyword search: "specific phrase"
- search-graph       # Relationship discovery: find connected notes
- search-combined    # Hybrid search combining all strategies
- read-note          # Read specific files with metadata
- list-vaults        # Vault management and statistics
- health-status      # System health monitoring
- performance-metrics # Performance analytics and optimization
```

**Example interaction in Claude Desktop:**
> "Find notes about machine learning algorithms"
> 
> *Uses semantic search to find related concepts like neural networks, deep learning, AI, even if those exact words aren't in the query*

## Claude Desktop Integration

Add to your Claude Desktop configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio",
      "args": ["/path/to/your/obsidian/vault"],
      "type": "stdio",
      "cwd": "/path/to/jarvis-assistant"
    }
  }
}
```

*Replace paths with your actual directories*

## Architecture Overview

```
Claude Desktop → MCP Server → Search Services → Databases
                              ├── Vector Search (DuckDB)
                              ├── Graph Search (Neo4j)
                              └── Vault Access (File System)
```

## Common Issues

- **Binary not found**: Run `uv sync` to create the virtual environment
- **Permission denied**: On Unix: `chmod +x .venv/bin/jarvis-mcp-stdio`
- **Vault not found**: Use absolute paths only
- **Server won't start**: Test with `uv run jarvis --help`

## Documentation

📚 **Complete Documentation**: [docs/README.md](docs/README.md)

### Quick Links
- **[Project Overview](docs/01-overview/project-overview.md)** - What is Jarvis Assistant?
- **[Quick Start Guide](docs/03-getting-started/quick-start.md)** - 5-minute setup
- **[Common Workflows](docs/04-usage/common-workflows.md)** - Typical usage patterns
- **[API Reference](docs/06-reference/api-reference.md)** - Complete MCP tool documentation
- **[Troubleshooting](docs/07-maintenance/troubleshooting.md)** - Common issues and solutions

### For Different Audiences
- **New Users**: Start with [Project Overview](docs/01-overview/project-overview.md)
- **Developers**: See [Developer Guide](docs/05-development/developer-guide.md)
- **AI Tools**: Reference [System Design](docs/02-system-design/data-flow.md)

## Contributing

1. Read the [Developer Guide](docs/05-development/developer-guide.md)
2. Follow the [Code Standards](docs/05-development/code-standards.md)
3. Add tests for new functionality
4. Update documentation for changes

## License

MIT License - see LICENSE file for details.

---

**Status**: Production ready with 8 working MCP tools  
**Next**: Enhanced performance and additional search capabilities  
**Support**: See [troubleshooting guide](docs/07-maintenance/troubleshooting.md) or open an issue