#!/usr/bin/env python3
"""
MCP Server entry point with proper stdio handling.

This module provides a clean entry point for the MCP server that
ensures logging goes to stderr while MCP protocol uses stdout.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from jarvis.mcp.server import run_mcp_server
from jarvis.utils.config import get_settings
from jarvis.utils.logging import configure_root_logging


def setup_mcp_logging():
    """Setup logging for MCP server - logs to stderr only."""
    settings = get_settings()
    log_file = settings.get_log_file_path()

    # Centralized logging configuration (stderr + optional file)
    # Respect settings.log_level and avoid duplicate handlers
    configure_root_logging(
        level=settings.log_level,
        structured=False,
        log_file=log_file,
    )

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
            # Use settings to get database path
            from jarvis.utils.config import get_settings
            settings = get_settings()
            db_path = settings.get_database_path()

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup vault configuration
        vaults = {"default": vault_path}

        logger.info(f"Starting MCP server with vault: {vault_path}, database: {db_path}")

        # Get settings
        settings = get_settings()

        # Run the MCP server (enable watch mode from settings)
        await run_mcp_server(vaults, db_path, settings, watch=settings.vault_watch)

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)

def main_sync():
    """Synchronous entry point for scripts."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()
