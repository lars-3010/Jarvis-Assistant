#!/bin/bash
# MCP Server Testing Commands
# 
# This script contains various commands used to test and debug the MCP server
# during the development process.

# Load environment variables
source .env 2>/dev/null || echo "Warning: .env file not found"

echo "=== MCP Server Testing Commands ==="

# 1. Test MCP server with direct protocol message
echo "1. Testing MCP server with protocol message:"
echo "Command:"
echo 'echo \'{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}\' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"'
echo ""

# 2. Test debug script
echo "2. Testing with debug script:"
echo "Command:"
echo "python3 resources/debug/debug_mcp.py"
echo ""

# 3. Test with environment variables
echo "3. Testing with environment variables:"
echo "Command:"
echo 'python3 resources/debug/mcp_wrapper.py'
echo ""

# 4. Test original jarvis mcp command
echo "4. Testing original jarvis mcp command:"
echo "Command:"
echo 'uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"'
echo ""

# 5. Check MCP server logs
echo "5. Check MCP server logs:"
echo "Command:"
echo "tail -f ~/.jarvis/mcp_server.log"
echo ""

# 6. Test MCP server availability
echo "6. Test MCP server tools:"
echo "Command:"
echo 'echo \'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}\' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"'
echo ""

# 7. Test semantic search
echo "7. Test semantic search:"
echo "Command:"
echo 'echo \'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search-semantic", "arguments": {"query": "machine learning", "limit": 5}}}\' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"'
echo ""

# 8. Test vault listing
echo "8. Test vault listing:"
echo "Command:"
echo 'echo \'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list-vaults", "arguments": {}}}\' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"'
echo ""

echo "=== End of Testing Commands ==="