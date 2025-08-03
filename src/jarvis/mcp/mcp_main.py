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