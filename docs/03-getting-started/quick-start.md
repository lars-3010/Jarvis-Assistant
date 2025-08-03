# Quick Start Guide

*Get Jarvis Assistant running in 5 minutes*

## Prerequisites

- **Python 3.11+** installed on your system
- **Obsidian vault** with some notes
- **Claude Desktop** (for MCP integration)

## 1. Install UV Package Manager

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

## 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd jarvis-assistant

# Install dependencies (creates virtual environment automatically)
uv sync

# Verify installation
uv run jarvis --help
```

## 3. Index Your Vault

```bash
# Replace with your actual vault path
uv run jarvis index --vault /path/to/your/obsidian/vault

# Example for macOS iCloud vault:
# uv run jarvis index --vault "/Users/username/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault"
```

Expected output:
```
Indexing vault: /path/to/your/obsidian/vault
Found 150 markdown files
Generating embeddings... ████████████████████████████████ 100%
Vector index created successfully
Processed 150 files in 45.2s
```

## 4. Test Semantic Search

```bash
# Test the search functionality
uv run jarvis search --vault /path/to/your/vault --query "productivity tips"
```

Expected output:
```
Semantic search results for: "productivity tips"

1. /Daily Notes/2024-01-15.md (Score: 0.89)
   Today I learned about the Pomodoro Technique for better productivity...

2. /Projects/GTD System.md (Score: 0.82)
   Getting Things Done methodology for organizing tasks and projects...

Found 5 results in 0.3s
```

## 5. Start MCP Server

```bash
# Start the MCP server for Claude Desktop
uv run jarvis mcp --vault /path/to/your/vault --watch

# You should see:
# MCP server started for vault: /path/to/your/vault
# Listening on stdio for Claude Desktop
# Press Ctrl+C to stop
```

## 6. Configure Claude Desktop

Create or edit `~/.claude.json`:

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

**Replace the paths:**
- `/path/to/jarvis-assistant` → Your actual project directory
- `/path/to/your/obsidian/vault` → Your actual vault path

## 7. Test in Claude Desktop

1. **Restart Claude Desktop** after updating configuration
2. **Open a new conversation**
3. **Test the integration:**

```
You: "Search my vault for notes about machine learning"
Claude: [Uses search-semantic tool to find relevant notes]

You: "Read the note about neural networks"
Claude: [Uses read-note tool to access specific content]
```

## Quick Troubleshooting

### Binary Not Found
```bash
# Check if virtual environment was created
ls -la .venv/bin/jarvis-mcp-stdio

# If missing, reinstall
uv sync
```

### Permission Denied (Unix/macOS)
```bash
# Fix permissions
chmod +x .venv/bin/jarvis-mcp-stdio
```

### Vault Path Issues
```bash
# Use absolute paths only
pwd  # Get current directory
ls "/full/path/to/your/vault"  # Verify vault exists
```

### Claude Desktop Not Connecting
1. Check Claude Desktop logs for errors
2. Verify paths in `~/.claude.json` are absolute
3. Test server manually: `uv run jarvis mcp --vault /path --verbose`

## Next Steps

### For Basic Usage
- **[Common Workflows](../04-usage/common-workflows.md)** - Typical usage patterns
- **[API Examples](../04-usage/api-examples.md)** - Copy-paste examples for all tools

### For Advanced Setup
- **[Detailed Installation](detailed-installation.md)** - Complete setup options
- **[Configuration Guide](configuration.md)** - Customize settings

### For Development
- **[Developer Guide](../05-development/developer-guide.md)** - Contributing to the project
- **[Testing Guide](../05-development/testing-strategy.md)** - Running tests

## Available MCP Tools

Once configured, Claude Desktop has access to these tools:

| Tool | Description | Example Use |
|------|-------------|-------------|
| `search-semantic` | Find related content by meaning | "Find notes about productivity" |
| `search-vault` | Keyword search | "Find notes containing 'docker'" |
| `search-graph` | Discover relationships | "What's connected to this note?" |
| `read-note` | Read specific files | "Read my daily note from yesterday" |
| `list-vaults` | Vault management | "Show vault statistics" |

## Common First Commands

```bash
# Get help
uv run jarvis --help

# Check system status
uv run jarvis stats --vault /path/to/vault

# Re-index after adding new notes
uv run jarvis index --vault /path/to/vault --force

# Test specific search
uv run jarvis search --vault /path/to/vault --query "your search here"

# Run with verbose logging
uv run jarvis --verbose mcp --vault /path/to/vault
```

## Performance Tips

- **SSD recommended** for better database performance
- **4GB+ RAM** for optimal embedding generation
- **Index regularly** after adding many new notes
- **Use specific queries** for better semantic search results

## Success Indicators

✅ **Installation successful**: `uv run jarvis --help` shows available commands  
✅ **Indexing works**: See progress bar and "Vector index created successfully"  
✅ **Search works**: `uv run jarvis search` returns relevant results  
✅ **MCP server starts**: No errors when running `uv run jarvis mcp`  
✅ **Claude integration**: Claude Desktop can use the 5 MCP tools  

---

**Estimated time**: 5-10 minutes for basic setup  
**Next**: [Detailed Installation Guide](detailed-installation.md) for advanced options  
**Help**: [Troubleshooting Guide](../07-maintenance/troubleshooting.md) for common issues