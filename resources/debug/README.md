# MCP Server Debugging Archive

This directory contains all the debugging files and documentation used to solve the MCP server integration issues with Claude Code.

## Files

### Documentation
- **`mcp_debugging_process.md`** - Complete debugging process documentation
- **`README.md`** - This file

### Debug Scripts
- **`debug_mcp.py`** - Interactive script to test MCP server startup and protocol communication
- **`mcp_wrapper.py`** - Wrapper script with logging redirection (deprecated approach)
- **`testing_commands.sh`** - Shell script with various testing commands

## Problem Summary

The main issue was that the MCP server was using `stdout` for logging, which interfered with the MCP JSON-RPC protocol communication. The solution was to create a dedicated MCP entry point (`src/jarvis/mcp/mcp_main.py`) that properly redirects all logging to `stderr` and log files.

## Key Files Created in Main Project

1. **`src/jarvis/mcp/mcp_main.py`** - Clean MCP entry point with proper stdio handling
2. **`jarvis-mcp-stdio`** script entry in `pyproject.toml`

## How to Use Debug Scripts

### 1. Test MCP Protocol Communication
```bash
python3 resources/debug/debug_mcp.py
```

### 2. Manual MCP Server Test
```bash
# Load environment variables first
source .env
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"
```

### 3. View All Testing Commands
```bash
./resources/debug/testing_commands.sh
```

## Final Working Configuration

### Claude Code MCP Configuration
```json
"jarvis": {
  "command": "uv",
  "args": ["run", "jarvis-mcp-stdio", "${JARVIS_VAULT_PATH}"],
  "cwd": "/path/to/your/Jarvis-Assistant",
  "env": {
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "JARVIS_VAULT_PATH": "/path/to/your/obsidian/vault"
  }
}
```

**Note**: Replace `/path/to/your/Jarvis-Assistant` and `/path/to/your/obsidian/vault` with your actual paths, or ensure your `.env` file is properly configured.

## Logs Location

- **MCP Server Logs**: `~/.jarvis/mcp_server.log`
- **Database**: `~/.jarvis/jarvis.duckdb`

## Success Indicators

When the MCP server is working correctly, you should see:
- Clean JSON-RPC responses on stdout
- All logging messages in stderr and log files
- Proper MCP protocol initialization responses
- Available tools: `search-semantic`, `read-note`, `list-vaults`, `search-vault`