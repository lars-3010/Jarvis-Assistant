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
import logging
import logging.config
from jarvis.utils.config import JarvisSettings, get_settings

# Redirect logging to a file to avoid interfering with MCP stdio
# Use settings to get log file path
settings = get_settings()
log_file = settings.get_log_file_path()

# Centralized logging: logs to file + stderr; MCP stdio remains on stdout
cfg = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "standard", "stream": "ext://sys.stderr"},
        "file": {"class": "logging.FileHandler", "level": "INFO", "formatter": "standard", "filename": str(log_file)},
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}
logging.config.dictConfig(cfg)
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
            # Use settings to get database path
            db_path = settings.get_database_path()
        
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
