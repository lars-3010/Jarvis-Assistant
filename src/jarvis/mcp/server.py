"""
MCP server implementation for Jarvis Assistant.

This module implements the Model Context Protocol server that exposes
semantic search and graph search capabilities to Claude Desktop.
"""

import asyncio
import json
import os
import shutil
import signal
import uuid
from pathlib import Path

# Neo4j exception handling
import mcp.server.stdio
from jarvis.mcp.container_context import ContainerAwareMCPServerContext
from jarvis.services.database_initializer import DatabaseInitializer
from jarvis.utils.config import JarvisSettings, get_settings
from jarvis.utils.database_errors import DatabaseError, DatabaseErrorHandler
from jarvis.utils.errors import JarvisError, ServiceUnavailableError, ToolExecutionError
import logging
import logging.config
from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

logger = logging.getLogger(__name__)



def create_mcp_server(
    vaults: dict[str, Path],
    database_path: Path,
    settings: JarvisSettings | None = None
) -> Server:
    """Create and configure the MCP server.
    
    Args:
        vaults: Dictionary mapping vault names to paths
        database_path: Path to DuckDB database file
        settings: Optional settings override
        
    Returns:
        Configured MCP server instance
    """
    server = Server("jarvis-assistant")

    # Use container-aware MCP server context (modern architecture)
    logger.info("Using container-aware MCP server context")
    context = ContainerAwareMCPServerContext(vaults, database_path, settings)

    # Initialize event system for reactive analytics
    from jarvis.core.event_integration import get_event_integration_manager
    from jarvis.core.events import get_event_bus

    logger.info("🎯 Initializing event system")
    event_bus = get_event_bus(metrics=context.metrics)
    event_integration_manager = get_event_integration_manager(metrics=context.metrics)

    # Start event system (background task)
    async def start_event_system():
        try:
            await event_bus.start()
            await event_integration_manager.start()
            logger.info("✅ Event system started successfully")
        except Exception as e:
            logger.error(f"❌ Event system startup failed: {e}")

    # Schedule event system startup
    import asyncio
    asyncio.create_task(start_event_system())

    # Initialize plugin registry with automatic discovery
    from jarvis.core.interfaces import (
        IGraphDatabase,
        IHealthChecker,
        IMetrics,
        IVaultReader,
        IVectorSearcher,
    )
    from jarvis.mcp.plugins.discovery import PluginDiscovery
    from jarvis.mcp.plugins.registry import PluginRegistry

    # Use the dependency injection container for plugins
    container_for_plugins = context.container

    # Initialize plugin registry and discovery system
    plugin_registry = PluginRegistry(container_for_plugins)
    plugin_discovery = PluginDiscovery(plugin_registry)

    # Auto-discover and load all plugins (built-in + external)
    discovery_stats = plugin_discovery.discover_and_load(include_builtin=True)
    logger.info(f"Plugin discovery completed: {discovery_stats['discovery']['plugins_registered']} plugins registered, {discovery_stats['loading']['plugins_loaded']} plugins loaded")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available MCP tools from plugin registry."""
        corr = str(uuid.uuid4())
        logger.info(f"[corr={corr}] list_tools called")
        tools = plugin_registry.get_tool_definitions()
        logger.info(f"[corr={corr}] list_tools returning {len(tools)} tools")
        return tools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests via plugin registry."""
        corr = str(uuid.uuid4())
        logger.info(f"[corr={corr}] call_tool start: name={name}")
        # Check cache first
        if context.mcp_cache:
            cached_results = context.mcp_cache.get(name, arguments or {})
            if cached_results:
                logger.debug(f"Returning cached results for tool: {name}")
                return cached_results

        try:
            # Delegate to plugin registry
            results = await plugin_registry.execute_tool(name, arguments or {})

            # Cache results if successful
            if context.mcp_cache:
                context.mcp_cache.put(name, arguments or {}, results)

            logger.info(f"[corr={corr}] call_tool success: name={name}, items={len(results)}")
            return results
        except DatabaseError as db_error:
            logger.error(f"[corr={corr}] Database error in tool {name}: {db_error}")

            # Format enhanced database error for MCP response
            error_handler = DatabaseErrorHandler(context.database_path)
            formatted_error = error_handler.format_error_for_user(db_error)

            return [
                types.TextContent(
                    type="text",
                    text=f"Database Error in {name}:\n\n{formatted_error}"
                )
            ]
        except ToolExecutionError as e:
            logger.error(f"[corr={corr}] Tool execution error in {name}: {e}")

            # Provide enhanced error information if available
            error_text = f"❌ Error executing {name}: {e!s}"

            if hasattr(e, 'suggestions') and e.suggestions:
                error_text += "\n\n💡 Troubleshooting Steps:"
                for i, suggestion in enumerate(e.suggestions, 1):
                    error_text += f"\n   {i}. {suggestion}"

            return [
                types.TextContent(
                    type="text",
                    text=error_text
                )
            ]
        except JarvisError as e:
            logger.error(f"[corr={corr}] Jarvis error in tool {name}: {e}")

            # Provide enhanced error information if available
            error_text = f"❌ Internal error in {name}: {e!s}"

            if hasattr(e, 'suggestions') and e.suggestions:
                error_text += "\n\n💡 Troubleshooting Steps:"
                for i, suggestion in enumerate(e.suggestions, 1):
                    error_text += f"\n   {i}. {suggestion}"
            elif hasattr(e, 'error_code'):
                error_text += f"\n\n🔍 Error Code: {e.error_code}"
                error_text += "\n💡 This error may be temporary - try again in a moment"

            return [
                types.TextContent(
                    type="text",
                    text=error_text
                )
            ]
        except Exception as e:
            logger.error(f"[corr={corr}] Unexpected error in tool {name}: {e}")

            # Provide basic troubleshooting guidance
            error_text = f"❌ An unexpected error occurred in {name}: {e!s}"
            error_text += "\n\n💡 Troubleshooting Steps:"
            error_text += "\n   1. Try the operation again in a moment"
            error_text += "\n   2. Check if the vault path is accessible"
            error_text += "\n   3. Verify the database is not being used by another process"
            error_text += "\n   4. Check system logs for additional error details"

            return [
                types.TextContent(
                    type="text",
                    text=error_text
                )
            ]

    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        """List available resources."""
        resources = []

        # Add vault resources
        for vault_name, vault_path in context.vaults.items():
            resources.append(
                types.Resource(
                    uri=f"jarvis://vault/{vault_name}",
                    name=f"Vault: {vault_name}",
                    description=f"Obsidian vault at {vault_path}",
                    mimeType="application/json"
                )
            )

        # Add database resource
        resources.append(
            types.Resource(
                uri="jarvis://database/stats",
                name="Vector Database Statistics",
                description="Statistics and information about the vector database",
                mimeType="application/json"
            )
        )

        return resources

    @server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Read a resource."""
        try:
            if uri.startswith("jarvis://vault/"):
                vault_name = uri.replace("jarvis://vault/", "")
                if vault_name in context.vaults:
                    stats = context.searcher.get_vault_stats()
                    vault_stat = stats.get(vault_name, {})
                    return json.dumps(vault_stat, indent=2)
                else:
                    raise ValueError(f"Unknown vault: {vault_name}")
            elif uri == "jarvis://database/stats":
                model_info = context.searcher.get_model_info()
                return json.dumps(model_info, indent=2)
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return f"Error reading resource: {e!s}"

    return server


async def run_mcp_server(
    vaults: dict[str, Path],
    database_path: Path,
    settings: JarvisSettings | None = None,
    watch: bool = False
) -> None:
    """Run the MCP server with stdio transport.
    
    Args:
        vaults: Dictionary mapping vault names to paths
        database_path: Path to DuckDB database file
        settings: Optional settings override
        watch: Whether to enable file watching for automatic reindexing
    """
    logger.info(f"🚀 Starting MCP server with {len(vaults)} vaults, watch={watch}")
    logger.debug(f"📁 Vaults: {[(name, str(path)) for name, path in vaults.items()]}")
    logger.debug(f"💾 Database: {database_path}")

    # Initialize database using DatabaseInitializer with enhanced error handling
    try:
        logger.info("💾 Initializing database for MCP server")
        initializer = DatabaseInitializer(database_path, settings or get_settings())
        error_handler = DatabaseErrorHandler(database_path)

        if not initializer.ensure_database_exists():
            logger.error("❌ Database initialization failed")

            # Get detailed database information for troubleshooting
            try:
                db_info = initializer.get_database_info()
                if db_info.get('error_message'):
                    logger.error(f"❌ Database error: {db_info['error_message']}")
            except Exception:
                pass

            # Provide enhanced error guidance
            logger.error("❌ The database could not be created or accessed")
            logger.error("💡 Enhanced troubleshooting steps:")
            logger.error(f"💡   Database path: {database_path}")
            logger.error(f"💡   Parent directory exists: {database_path.parent.exists()}")
            logger.error(f"💡   Parent directory writable: {os.access(database_path.parent, os.W_OK) if database_path.parent.exists() else 'N/A'}")

            # Check disk space
            try:
                disk_usage = shutil.disk_usage(database_path.parent)
                available_mb = disk_usage.free / (1024 * 1024)
                logger.error(f"💡   Available disk space: {available_mb:.1f} MB")
                if available_mb < 10:
                    logger.error("💡   ⚠️ Low disk space detected - this may be the cause")
            except Exception:
                logger.error("💡   Could not check disk space")

            logger.error("💡 Common solutions:")
            logger.error(f"💡   1. Check file permissions: ls -la '{database_path.parent}'")
            logger.error(f"💡   2. Create directory: mkdir -p '{database_path.parent}'")
            logger.error("💡   3. Check disk space: df -h")
            logger.error("💡   4. Try manual database creation: jarvis init-database")

            raise ServiceUnavailableError("Database initialization failed - cannot start MCP server")

        # Get database information for logging
        db_info = initializer.get_database_info()
        logger.info(f"✅ Database ready: {db_info['note_count']} notes, {db_info['size_mb']} MB")
        if db_info.get('has_embeddings'):
            logger.info(f"🧠 Embeddings available: {db_info['embedding_count']} notes with vectors")
        else:
            logger.warning("⚠️ No embeddings found - semantic search may not work until indexing is complete")
            logger.info("💡 Run 'jarvis index' to generate embeddings for semantic search")

    except ServiceUnavailableError:
        # Re-raise service unavailable errors as-is
        raise
    except DatabaseError as db_error:
        # Handle enhanced database errors with user-friendly messaging
        logger.error(f"💥 Database error: {db_error}")

        # Log suggestions from enhanced error handling
        for suggestion in db_error.suggestions:
            logger.error(f"💡 {suggestion}")

        # Log additional context if available
        if db_error.context:
            if db_error.context.get('database_path'):
                logger.error(f"💡 Database path: {db_error.context['database_path']}")
            if db_error.context.get('permissions_info'):
                perm_info = db_error.context['permissions_info']
                logger.error(f"💡 Permissions - Parent readable: {perm_info.get('parent_readable', 'Unknown')}, writable: {perm_info.get('parent_writable', 'Unknown')}")

        raise ServiceUnavailableError(f"Database initialization failed: {db_error}") from db_error
    except Exception as db_init_error:
        # Handle generic database initialization errors
        logger.error(f"💥 Unexpected database initialization error: {db_init_error}")

        # Try to provide enhanced error information
        try:
            error_handler = DatabaseErrorHandler(database_path)
            enhanced_error = error_handler.handle_generic_database_error(db_init_error, "initialize database")

            logger.error("💡 Enhanced troubleshooting information:")
            for suggestion in enhanced_error.suggestions:
                logger.error(f"💡   {suggestion}")

        except Exception:
            # Fallback to basic error handling
            logger.error("💡 Basic troubleshooting steps:")
            logger.error("💡   1. Verify database path is accessible")
            logger.error("💡   2. Check file system permissions")
            logger.error("💡   3. Ensure no other processes are using the database")
            logger.error("💡   4. Try manual database creation with 'jarvis init-database'")

        raise ServiceUnavailableError(f"Database initialization failed: {db_init_error}") from db_init_error

    logger.debug("🚀 Creating MCP server")
    server = create_mcp_server(vaults, database_path, settings)

    # Initialize file watcher if requested
    vector_worker = None
    if watch:
        logger.info("🔍 Watch mode enabled - setting up file monitoring")
        try:
            # Validate vault paths before starting watcher
            for vault_name, vault_path in vaults.items():
                if not vault_path.exists():
                    logger.error(f"❌ Vault path does not exist: {vault_name} -> {vault_path}")
                    logger.error("❌ File watching disabled due to invalid vault path")
                    watch = False
                    break
                elif not vault_path.is_dir():
                    logger.error(f"❌ Vault path is not a directory: {vault_name} -> {vault_path}")
                    logger.error("❌ File watching disabled due to invalid vault path")
                    watch = False
                    break
                else:
                    logger.debug(f"✅ Vault path validated: {vault_name} -> {vault_path}")
                    # Log special iCloud handling
                    if "iCloud" in str(vault_path):
                        logger.info(f"📱 iCloud vault detected: {vault_name} -> {vault_path}")
                        logger.info("📱 iCloud sync may cause additional file system events")

            if watch:  # Only proceed if all vaults are valid
                logger.info("🔧 Setting up file watching with shared database approach")
                try:
                    # DuckDB doesn't allow multiple connections with different configurations
                    # So we'll use a different approach: create a separate database path for the worker
                    # or use a shared connection approach

                    worker_db_path = database_path.parent / f"{database_path.stem}-worker{database_path.suffix}"
                    logger.debug(f"💾 Worker will use separate database: {worker_db_path}")

                    # Copy the existing database to the worker database if it exists
                    if database_path.exists() and not worker_db_path.exists():
                        logger.debug("📋 Copying main database to worker database")
                        shutil.copy2(database_path, worker_db_path)
                        logger.debug("✅ Database copied successfully")

                    # Initialize worker database using DatabaseInitializer
                    logger.debug(f"💾 Initializing worker database at {worker_db_path}")
                    worker_initializer = DatabaseInitializer(worker_db_path, settings or get_settings())

                    if not worker_initializer.ensure_database_exists():
                        logger.error("❌ Worker database initialization failed")
                        logger.error("❌ File watching cannot be enabled without a working database")
                        watch = False
                        logger.warning("⚠️ Disabling file watching due to database initialization failure")
                    else:
                        # Create database connection after successful initialization
                        worker_database = VectorDatabase(worker_db_path, read_only=False, create_if_missing=True)
                        logger.debug("💾 Worker database connection created successfully")

                        # Get worker database info for logging
                        worker_db_info = worker_initializer.get_database_info()
                        logger.debug(f"💾 Worker database ready: {worker_db_info['note_count']} notes, {worker_db_info['size_mb']} MB")

                        # Continue with vector worker setup
                        logger.info("🤖 Initializing vector encoder")
                        encoder = VectorEncoder()
                        logger.debug("🧠 Vector encoder initialized")

                        logger.info("👷 Creating vector worker")
                        # Create and start the vector worker
                        vector_worker = VectorWorker(
                            database=worker_database,
                            encoder=encoder,
                            vaults=vaults,
                            enable_watching=True,
                            auto_index=False  # Don't auto-index on startup
                        )
                        logger.debug("👷 Vector worker created successfully")

                        logger.info("🏃 Starting vector worker")
                        vector_worker.start()
                        logger.info(f"✅ File watching enabled for {len(vaults)} vaults")

                        # Log watcher status
                        stats = vector_worker.get_stats()
                        logger.info(f"📊 Worker stats: {stats['watchers_active']} watchers active, queue size: {stats['queue_size']}")

                        # Additional validation - check if workers started properly
                        if stats['watchers_active'] == 0:
                            logger.warning("⚠️ No file watchers started - file watching may not be working")
                            logger.warning("⚠️ This could be due to:")
                            logger.warning("⚠️   1. Invalid vault paths")
                            logger.warning("⚠️   2. Permission issues")
                            logger.warning("⚠️   3. File system monitoring not supported")
                            logger.warning("⚠️   4. iCloud sync conflicts")
                        else:
                            logger.info(f"✅ File watching setup successful - {stats['watchers_active']} watchers active")

                except Exception as db_error:
                    logger.error(f"💥 Database connection error: {db_error}")
                    logger.error(f"🔍 Exception type: {type(db_error).__name__}")
                    import traceback
                    logger.error(f"🔍 Full traceback:\n{traceback.format_exc()}")

                    # Provide specific error guidance
                    error_str = str(db_error).lower()
                    if "database is locked" in error_str or "database locked" in error_str:
                        logger.error("🔒 Database is locked - this may indicate:")
                        logger.error("🔒   1. Another process is using the database")
                        logger.error("🔒   2. A previous process didn't shut down cleanly")
                        logger.error("🔒   3. File system permissions issue")
                    elif "permission" in error_str or "access" in error_str:
                        logger.error("🔐 Permission/access error - check file permissions")
                    elif "no such file" in error_str or "not found" in error_str:
                        logger.error("📁 Database file not found - may need to run indexing first")
                    elif "different configuration" in error_str or "configuration" in error_str:
                        logger.error("🔧 DuckDB configuration conflict - multiple connections to same file")
                        logger.error("🔧 This is a known limitation with DuckDB file connections")

                    # Try to clean up
                    try:
                        if 'worker_database' in locals():
                            worker_database.close()
                    except Exception as cleanup_error:
                        logger.error(f"🧹 Error during database cleanup: {cleanup_error}")

                    watch = False
                    logger.warning("⚠️ Disabling file watching due to database connection issues")

        except Exception as e:
            logger.error(f"💥 Failed to start file watching: {e}")
            logger.error(f"🔍 Exception type: {type(e).__name__}")
            logger.error(f"📋 Full exception details: {e!s}")
            import traceback
            logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")

            # Clean up any partial initialization
            if vector_worker:
                try:
                    logger.info("🧹 Cleaning up partially initialized worker")
                    vector_worker.stop()
                except Exception as cleanup_error:
                    logger.error(f"🧹 Error during cleanup: {cleanup_error}")

            # Clean up database connection if it exists
            if 'worker_database' in locals():
                try:
                    logger.info("🧹 Cleaning up worker database connection")
                    worker_database.close()
                    logger.debug("✅ Worker database connection closed")
                except Exception as db_cleanup_error:
                    logger.error(f"🧹 Error cleaning up database: {db_cleanup_error}")

            # Clean up worker database file if it exists
            if 'worker_db_path' in locals() and worker_db_path.exists():
                try:
                    logger.info("🧹 Cleaning up worker database file")
                    worker_db_path.unlink()
                    logger.debug("✅ Worker database file removed")
                except Exception as file_cleanup_error:
                    logger.error(f"🧹 Error cleaning up worker database file: {file_cleanup_error}")

            vector_worker = None
            logger.warning("⚠️ Continuing without file watching")

    try:
        logger.info("🌊 Starting MCP server with stdio transport")

        # Set up signal handling for graceful shutdown
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown")
            shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("📡 MCP server stdio streams established")

            if shutdown_requested:
                logger.info("📡 Shutdown requested before server start")
                return

            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="jarvis-assistant",
                    server_version="0.2.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("⌨️ Keyboard interrupt received - shutting down gracefully")
        raise
    except Exception as e:
        logger.error(f"💥 MCP server error: {e}")
        logger.error(f"🔍 Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Clean up the vector worker
        if vector_worker:
            try:
                logger.info("🛑 Stopping vector worker")
                worker_stats = vector_worker.get_stats()
                logger.info(f"📊 Final worker stats: processed={worker_stats['files_processed']}, failed={worker_stats['files_failed']}, uptime={worker_stats['uptime']:.2f}s")

                vector_worker.stop()
                logger.info("✅ File watching stopped successfully")

                # Clean up worker database file
                worker_db_path = database_path.parent / f"{database_path.stem}-worker{database_path.suffix}"
                if worker_db_path.exists():
                    try:
                        logger.info("🧹 Cleaning up worker database file")
                        worker_db_path.unlink()
                        logger.debug("✅ Worker database file removed")
                    except Exception as file_cleanup_error:
                        logger.error(f"🧹 Error cleaning up worker database file: {file_cleanup_error}")

            except Exception as e:
                logger.error(f"💥 Error stopping file watching: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
        else:
            logger.debug("🚫 No vector worker to stop")

        logger.info("🏁 MCP server shutdown complete")


def main() -> None:
    """Main entry point for MCP server."""
    import os
    import sys

    # Default configuration for standalone execution
    default_vaults = {}

    # Check for database path from environment
    db_env = os.getenv("JARVIS_DATABASE_PATH")
    default_db_path = Path(db_env).expanduser() if db_env else Path.home() / ".jarvis" / "jarvis.duckdb"

    # Check for environment variables
    vault_env = os.getenv("JARVIS_VAULT_PATH")
    if vault_env:
        vault_path = Path(vault_env)
        if vault_path.exists():
            default_vaults["default"] = vault_path
            logger.info(f"Using vault from environment: {vault_path}")

    db_env = os.getenv("JARVIS_DB_PATH")
    if db_env:
        default_db_path = Path(db_env)
        logger.info(f"Using database from environment: {default_db_path}")

    if not default_vaults:
        logger.error("No vault configured. Set JARVIS_VAULT_PATH environment variable.")
        sys.exit(1)

    # Configure root logging for standalone run
    try:
        level = os.getenv("JARVIS_LOG_LEVEL", "INFO")
        cfg = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
            "handlers": {"console": {"class": "logging.StreamHandler", "level": level, "formatter": "standard", "stream": "ext://sys.stderr"}},
            "root": {"level": level, "handlers": ["console"]},
        }
        logging.config.dictConfig(cfg)
    except Exception:
        pass

    # Run the server
    asyncio.run(run_mcp_server(default_vaults, default_db_path))


if __name__ == "__main__":
    main()
