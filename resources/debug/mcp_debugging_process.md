# MCP Server Debugging Process

## Problem Description
The Jarvis MCP server was failing to start in Claude Code with the error:
```
MCP server "jarvis": Connection failed: McpError: MCP error -32000: Connection closed
```

## Root Cause Analysis

### 1. Initial Investigation
The MCP server was working when run manually via `uv run jarvis mcp --vault "..."` but failing when started by Claude Code.

### 2. Key Discovery
The issue was that the MCP server was using `sys.stdout` for logging, which interferes with the MCP protocol that requires clean stdout for JSON-RPC communication.

**Location of the issue:**
- `src/jarvis/utils/logging.py:64` - Console handler using `sys.stdout`
- `src/jarvis/utils/logging.py:109` - Root logger configuration using stdout

### 3. MCP Protocol Requirements
- **stdout**: Must be reserved for JSON-RPC MCP protocol messages only
- **stderr**: Should be used for all logging and debug output
- **stdin**: Used for receiving MCP protocol messages

## Debugging Steps Taken

### Step 1: Created Debug Script
Created `debug_mcp.py` to test MCP server startup and protocol communication:

```python
#!/usr/bin/env python3
"""Debug script to test MCP server startup."""

import sys
import os
import subprocess
import json

def test_mcp_server():
    """Test the MCP server startup process."""
    print("Testing MCP server startup...", file=sys.stderr)
    
    # Change to the project directory
    project_dir = "/path/to/your/Jarvis-Assistant"
    os.chdir(project_dir)
    
    # Run the MCP server command
    cmd = [
        "uv", "run", "jarvis", "mcp", 
        "--vault", "${JARVIS_VAULT_PATH}"
    ]
    
    print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
    print(f"Working directory: {os.getcwd()}", file=sys.stderr)
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send an MCP initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "debug-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print(f"Sending init request: {json.dumps(init_request)}", file=sys.stderr)
        
        # Send the request
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Wait for response with timeout
        import select
        import time
        
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print(f"Process exited with code: {process.returncode}", file=sys.stderr)
                break
                
            # Check if there's output available
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                line = process.stdout.readline()
                if line:
                    print(f"Server response: {line.strip()}", file=sys.stderr)
                    break
        
        # Get any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"Server stderr: {stderr_output}", file=sys.stderr)
            
        # Clean up
        process.terminate()
        process.wait()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    test_mcp_server()
```

**Result**: This revealed that the server was outputting logs to stdout, contaminating the MCP protocol stream.

### Step 2: Created Wrapper Script
Created `mcp_wrapper.py` to redirect logging to stderr and files:

```python
#!/usr/bin/env python3
"""
MCP Server Wrapper for Claude Code integration.

This wrapper ensures proper stdio handling for MCP communication
by redirecting logs to a separate file and only allowing MCP 
protocol messages on stdout.
"""

import os
import sys
from pathlib import Path
import asyncio
from typing import Dict, Optional

from jarvis.mcp.server import run_mcp_server
from jarvis.utils.logging import setup_logging
from jarvis.utils.config import JarvisSettings, get_settings

# Redirect logging to a file to avoid interfering with MCP stdio
log_file = Path.home() / ".jarvis" / "mcp_server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

# Setup logging to file only
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        # Only add stderr handler for critical errors
        logging.StreamHandler(sys.stderr)
    ]
)

# Set stderr handler to only show critical errors
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.CRITICAL)
logging.getLogger().handlers = [
    logging.FileHandler(log_file),
    stderr_handler
]

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for MCP server wrapper."""
    try:
        # Get configuration from environment or defaults
        vault_path = os.getenv("JARVIS_VAULT_PATH")
        database_path = os.getenv("JARVIS_DATABASE_PATH")
        
        if not vault_path:
            logger.critical("JARVIS_VAULT_PATH environment variable not set")
            sys.exit(1)
        
        vault_path = Path(vault_path)
        if not vault_path.exists():
            logger.critical(f"Vault path does not exist: {vault_path}")
            sys.exit(1)
        
        if database_path:
            db_path = Path(database_path)
        else:
            db_path = Path.home() / ".jarvis" / "jarvis.duckdb"
        
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup vault configuration
        vaults = {"default": vault_path}
        
        logger.info(f"Starting MCP server with vault: {vault_path}, database: {db_path}")
        
        # Get settings
        settings = get_settings()
        
        # Run the MCP server
        await run_mcp_server(vaults, db_path, settings)
        
    except Exception as e:
        logger.critical(f"Failed to start MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

**Issue**: This approach had dependency issues when running outside the UV environment.

### Step 3: Final Solution - Dedicated MCP Entry Point
Created `src/jarvis/mcp/mcp_main.py` with proper stdio handling:

```python
#!/usr/bin/env python3
"""
MCP Server entry point with proper stdio handling.

This module provides a clean entry point for the MCP server that
ensures logging goes to stderr while MCP protocol uses stdout.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional

from jarvis.mcp.server import run_mcp_server
from jarvis.utils.config import JarvisSettings, get_settings

def setup_mcp_logging():
    """Setup logging for MCP server - logs to stderr only."""
    log_file = Path.home() / ".jarvis" / "mcp_server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger to use stderr and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)  # Use stderr for MCP
        ]
    )
    
    # Set specific loggers to use stderr
    for logger_name in ['jarvis', 'jarvis.mcp', 'jarvis.services']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Clear any existing handlers
        logger.addHandler(logging.StreamHandler(sys.stderr))
        logger.addHandler(logging.FileHandler(log_file))
        logger.setLevel(logging.INFO)
        logger.propagate = False

async def main():
    """Main entry point for MCP server."""
    # Setup logging first
    setup_mcp_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Get configuration from CLI arguments passed through environment
        vault_path = os.getenv("JARVIS_VAULT_PATH")
        database_path = os.getenv("JARVIS_DATABASE_PATH")
        
        if not vault_path:
            # Try to get from command line arguments
            if len(sys.argv) > 1:
                vault_path = sys.argv[1]
            else:
                logger.error("No vault path provided")
                sys.exit(1)
        
        vault_path = Path(vault_path)
        if not vault_path.exists():
            logger.error(f"Vault path does not exist: {vault_path}")
            sys.exit(1)
        
        if database_path:
            db_path = Path(database_path)
        else:
            db_path = Path.home() / ".jarvis" / "jarvis.duckdb"
        
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup vault configuration
        vaults = {"default": vault_path}
        
        logger.info(f"Starting MCP server with vault: {vault_path}, database: {db_path}")
        
        # Get settings
        settings = get_settings()
        
        # Run the MCP server
        await run_mcp_server(vaults, db_path, settings)
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)

def main_sync():
    """Synchronous entry point for scripts."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
```

## Testing Commands

### 1. Manual MCP Server Test
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | uv run jarvis-mcp-stdio "$JARVIS_VAULT_PATH"
```

### 2. Debug Script Test
```bash
python3 debug_mcp.py
```

### 3. Environment Variable Test
```bash
python3 mcp_wrapper.py
```

## Final Configuration

### pyproject.toml Addition
```toml
[project.scripts]
jarvis = "jarvis.main:main"
jarvis-mcp = "jarvis.mcp.server:main"
jarvis-mcp-stdio = "jarvis.mcp.mcp_main:main_sync"
```

### Claude Code MCP Configuration
```json
"jarvis": {
  "command": "uv",
  "args": ["run", "jarvis-mcp-stdio", "${JARVIS_VAULT_PATH}"],
  "cwd": "/path/to/your/Jarvis-Assistant",
  "env": {
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8"
  }
}
```

## Key Lessons Learned

1. **MCP Protocol Strictness**: MCP requires clean stdout for JSON-RPC communication
2. **Logging Separation**: All application logs must go to stderr or files, never stdout
3. **Environment Variables**: Use `PYTHONUNBUFFERED=1` for real-time output
4. **Testing Methodology**: Always test MCP servers with actual protocol messages
5. **Dedicated Entry Points**: Create separate entry points for different use cases (CLI vs MCP)

## Files Created During Debugging

1. `debug_mcp.py` - Debug script for testing MCP server startup
2. `mcp_wrapper.py` - Wrapper script with logging redirection
3. `src/jarvis/mcp/mcp_main.py` - Final MCP entry point with proper stdio handling

All debugging files are preserved in `resources/debug/` for future reference.