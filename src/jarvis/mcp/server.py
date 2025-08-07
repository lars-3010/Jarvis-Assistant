"""
MCP server implementation for Jarvis Assistant.

This module implements the Model Context Protocol server that exposes
semantic search and graph search capabilities to Claude Desktop.
"""

import asyncio
import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vector.searcher import VectorSearcher
from jarvis.services.vector.indexer import VectorIndexer
from jarvis.services.vector.worker import VectorWorker
from jarvis.services.vault.reader import VaultReader
from jarvis.services.graph.database import GraphDatabase
from jarvis.services.health import HealthChecker
from jarvis.services.ranking import ResultRanker
from jarvis.services.database_initializer import DatabaseInitializer
from jarvis.mcp.cache import MCPToolCache
from jarvis.mcp.container_context import ContainerAwareMCPServerContext
from jarvis.monitoring.metrics import JarvisMetrics
from jarvis.models.document import SearchResult
from jarvis.utils.logging import setup_logging
from jarvis.utils.config import JarvisSettings, get_settings
from jarvis.utils.errors import ToolExecutionError, ServiceUnavailableError, JarvisError
from jarvis.utils.database_errors import DatabaseErrorHandler, DatabaseError

# Neo4j exception handling
from neo4j.exceptions import Neo4jError

logger = setup_logging(__name__)


class MCPServerContext:
    """Context for MCP server operations."""
    
    def __init__(
        self,
        vaults: Dict[str, Path],
        database_path: Path,
        settings: Optional[JarvisSettings] = None
    ):
        """Initialize MCP server context.
        
        Args:
            vaults: Dictionary mapping vault names to paths
            database_path: Path to DuckDB database file
            settings: Optional settings override
        """
        self.vaults = vaults
        self.database_path = database_path
        self.settings = settings or get_settings()
        
        # Initialize services
        logger.debug(f"💾 Initializing MCP server context database (read-only): {database_path}")
        self.database = VectorDatabase(database_path, read_only=True)
        logger.debug(f"🧠 Initializing MCP server context encoder")
        self.encoder = VectorEncoder()
        logger.debug(f"🔍 Initializing MCP server context searcher")
        self.searcher = VectorSearcher(self.database, self.encoder, vaults)
        
        # Initialize graph database with error handling
        self.graph_database = GraphDatabase(self.settings)
        
        # Initialize vault readers
        self.vault_readers = {}
        for vault_name, vault_path in vaults.items():
            try:
                self.vault_readers[vault_name] = VaultReader(str(vault_path))
                logger.info(f"Initialized vault reader for {vault_name}")
            except Exception as e:
                logger.error(f"Failed to initialize vault reader for {vault_name}: {e}")
        
        # Initialize health checker
        self.health_checker = HealthChecker(self.settings)
        
        # Initialize result ranker
        self.ranker = ResultRanker()
        
        # Initialize MCP tool cache
        self.mcp_cache = MCPToolCache(self.settings.mcp_cache_size, self.settings.mcp_cache_ttl)
        
        # Initialize metrics collection
        self.metrics = JarvisMetrics() if self.settings.metrics_enabled else None
        
        logger.info(f"MCP server context initialized with {len(vaults)} vaults, metrics_enabled={self.settings.metrics_enabled}")
    
    def close(self):
        """Clean up resources."""
        logger.debug("🧹 Cleaning up MCP server context resources")
        
        if hasattr(self, 'database'):
            try:
                logger.debug("🧹 Closing vector database connection")
                self.database.close()
                logger.debug("✅ Vector database connection closed")
            except Exception as e:
                logger.error(f"💥 Error closing vector database: {e}")
        
        if hasattr(self, 'graph_database') and self.graph_database:
            try:
                logger.debug("🧹 Closing graph database connection")
                self.graph_database.close()
                logger.debug("✅ Graph database connection closed")
            except Exception as e:
                logger.error(f"💥 Error closing graph database: {e}")
        
        if hasattr(self, 'mcp_cache') and self.mcp_cache:
            try:
                logger.debug("🧹 Clearing MCP cache")
                self.mcp_cache.clear()
                logger.debug("✅ MCP cache cleared")
            except Exception as e:
                logger.error(f"💥 Error clearing MCP cache: {e}")
        
        logger.debug("✅ MCP server context cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        logger.debug("💪 Exiting MCP server context manager")
        if exc_type:
            logger.error(f"💥 Exception in context manager: {exc_type.__name__}: {exc_val}")
        self.close()
        return False  # Don't suppress exceptions

    def clear_cache(self):
        """Clears the MCP tool cache."""
        if self.mcp_cache:
            self.mcp_cache.clear()


def create_mcp_server(
    vaults: Dict[str, Path],
    database_path: Path,
    settings: Optional[JarvisSettings] = None
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
    
    # Use dependency injection container if enabled
    if settings and settings.use_dependency_injection:
        logger.info("Using container-aware MCP server context")
        context = ContainerAwareMCPServerContext(vaults, database_path, settings)
    else:
        logger.info("Using traditional MCP server context")
        context = MCPServerContext(vaults, database_path, settings)
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available MCP tools."""
        return [
            types.Tool(
                name="search-semantic",
                description="Perform semantic search across vault content using natural language queries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50, "description": "Maximum number of results to return"},
                        "vault": {"type": "string", "description": "Optional vault name to search within"},
                        "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Minimum similarity score (0.0-1.0)"}
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="read-note",
                description="Read the content of a specific note from a vault",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the note relative to vault root"},
                        "vault": {"type": "string", "description": "Vault name (uses first available if not specified)"}
                    },
                    "required": ["path"],
                },
            ),
            types.Tool(
                name="list-vaults",
                description="List all available vaults and their statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="search-vault",
                description="Search for files in vault by filename or content (traditional search)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term for filenames or content"},
                        "vault": {"type": "string", "description": "Vault name to search in"},
                        "search_content": {"type": "boolean", "default": False, "description": "Whether to search within file content"},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100, "description": "Maximum number of results"}
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="search-graph",
                description="Search for notes and their relationships in the knowledge graph.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query_note_path": {"type": "string", "description": "The path to the note to use as the center of the search."},
                        "depth": {"type": "integer", "default": 1, "description": "How many relationship levels to traverse."}
                    },
                    "required": ["query_note_path"],
                },
            ),
            types.Tool(
                name="search-combined",
                description="Perform a combined semantic and keyword search across vault content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50, "description": "Maximum number of results to return"},
                        "vault": {"type": "string", "description": "Optional vault name to search within"},
                        "search_content": {"type": "boolean", "default": True, "description": "Whether to include keyword search within file content."}
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get-health-status",
                description="Get the health status of all Jarvis Assistant services.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="get-performance-metrics",
                description="Get performance metrics and statistics for MCP tools and services.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reset_after_read": {"type": "boolean", "default": False, "description": "Whether to reset metrics after reading them"},
                        "filter_prefix": {"type": "string", "description": "Optional prefix to filter metrics (e.g., 'mcp_tool_' for tool metrics only)"}
                    },
                },
            ),
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests."""
        # Check cache first
        if context.mcp_cache:
            cached_results = context.mcp_cache.get(name, arguments or {})
            if cached_results:
                logger.debug(f"Returning cached results for tool: {name}")
                return cached_results

        try:
            results = []
            if name == "search-semantic":
                results = await _handle_semantic_search(context, arguments or {})
            elif name == "read-note":
                results = await _handle_read_note(context, arguments or {})
            elif name == "list-vaults":
                results = await _handle_list_vaults(context, arguments or {})
            elif name == "search-vault":
                results = await _handle_search_vault(context, arguments or {})
            elif name == "search-graph":
                results = await _handle_search_graph(context, arguments or {})
            elif name == "search-combined":
                results = await _handle_search_combined(context, arguments or {})
            elif name == "get-health-status":
                results = await _handle_get_health_status(context, arguments or {})
            elif name == "get-performance-metrics":
                results = await _handle_get_performance_metrics(context, arguments or {})
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            # Cache results if successful
            if context.mcp_cache:
                context.mcp_cache.put(name, arguments or {}, results)
            
            return results
        except DatabaseError as db_error:
            logger.error(f"Database error in tool {name}: {db_error}")
            
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
            logger.error(f"Tool execution error in {name}: {e}")
            
            # Provide enhanced error information if available
            error_text = f"❌ Error executing {name}: {str(e)}"
            
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
            logger.error(f"Jarvis error in tool {name}: {e}")
            
            # Provide enhanced error information if available
            error_text = f"❌ Internal error in {name}: {str(e)}"
            
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
            logger.error(f"Unexpected error in tool {name}: {e}")
            
            # Provide basic troubleshooting guidance
            error_text = f"❌ An unexpected error occurred in {name}: {str(e)}"
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
            return f"Error reading resource: {str(e)}"
    
    return server


async def _handle_semantic_search(
    context: MCPServerContext, 
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle semantic search requests."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)
    vault_name = arguments.get("vault")
    similarity_threshold = arguments.get("similarity_threshold")
    
    if not query:
        return [types.TextContent(type="text", text="Error: Query parameter is required")]
    
    try:
        # Record metrics for tool execution
        if context.metrics:
            with context.metrics.time_operation("mcp_tool_search_semantic"):
                # Perform semantic search
                results = context.searcher.search(
                    query=query,
                    top_k=limit,
                    vault_name=vault_name,
                    similarity_threshold=similarity_threshold
                )
        else:
            # Perform semantic search without metrics
            results = context.searcher.search(
                query=query,
                top_k=limit,
                vault_name=vault_name,
                similarity_threshold=similarity_threshold
            )
        
        # Record result metrics
        if context.metrics:
            context.metrics.safe_record("mcp_tool_search_semantic_results", len(results))
        
        if not results:
            return [types.TextContent(
                type="text", 
                text=f"No results found for query: '{query}'"
            )]
        
        # Format results
        response_lines = [f"Found {len(results)} results for '{query}':\n"]
        
        # Rank results before formatting
        ranked_results = context.ranker.rank_results(results)

        for i, result in enumerate(ranked_results, 1):
            score = f"{result.similarity_score:.3f}"
            response_lines.append(
                f"{i}. **{result.path}** (vault: {result.vault_name}, score: {score})"
            )
        
        return [types.TextContent(type="text", text="\n".join(response_lines))]
        
    except ServiceError as e:
        logger.error(f"Semantic search error: {e}")
        return [types.TextContent(type="text", text=f"Search error: {str(e)}")]


async def _handle_read_note(
    context: MCPServerContext, 
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle note reading requests."""
    path = arguments.get("path", "")
    vault_name = arguments.get("vault")
    
    if not path:
        return [types.TextContent(type="text", text="Error: Path parameter is required")]
    
    # Determine which vault to use
    if vault_name and vault_name in context.vault_readers:
        vault_reader = context.vault_readers[vault_name]
    elif vault_name:
        return [types.TextContent(type="text", text=f"Error: Unknown vault '{vault_name}'")]
    else:
        # Use first available vault
        if not context.vault_readers:
            return [types.TextContent(type="text", text="Error: No vaults available")]
        vault_reader = next(iter(context.vault_readers.values()))
    
    try:
        content, metadata = vault_reader.read_file(path)
        
        # Format response
        response = f"# {metadata['path']}\n\n"
        response += f"**Size:** {metadata['size']} bytes  \n"
        response += f"**Modified:** {metadata['modified_formatted']}  \n\n"
        response += "---\n\n"
        response += content
        
        return [types.TextContent(type="text", text=response)]
        
    except ServiceError as e:
        logger.error(f"Error reading note {path}: {e}")
        return [types.TextContent(type="text", text=f"Error reading note: {str(e)}")]


async def _handle_list_vaults(
    context: MCPServerContext, 
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle vault listing requests."""
    try:
        vault_stats = context.searcher.get_vault_stats()
        validation = context.searcher.validate_vaults()
        
        response_lines = ["# Available Vaults\n"]
        
        for vault_name, vault_path in context.vaults.items():
            stats = vault_stats.get(vault_name, {})
            is_valid = validation.get(vault_name, False)
            
            status = "✅ Available" if is_valid else "❌ Unavailable"
            note_count = stats.get('note_count', 0)
            
            response_lines.append(f"## {vault_name}")
            response_lines.append(f"- **Status:** {status}")
            response_lines.append(f"- **Path:** `{vault_path}`")
            response_lines.append(f"- **Notes:** {note_count}")
            
            if stats.get('latest_modified'):
                latest = datetime.fromtimestamp(stats['latest_modified']).isoformat()
                response_lines.append(f"- **Last Modified:** {latest}")
            
            response_lines.append("")
        
        # Add model info
        model_info = context.searcher.get_model_info()
        response_lines.append("## Search Configuration")
        response_lines.append(f"- **Model:** {model_info.get('encoder_info', {}).get('model_name', 'Unknown')}")
        response_lines.append(f"- **Device:** {model_info.get('encoder_info', {}).get('device', 'Unknown')}")
        response_lines.append(f"- **Total Notes:** {model_info.get('database_note_count', 0)}")
        
        return [types.TextContent(type="text", text="\n".join(response_lines))]
        
    except ServiceError as e:
        logger.error(f"Error listing vaults: {e}")
        return [types.TextContent(type="text", text=f"Error listing vaults: {str(e)}")]


async def _handle_search_vault(
    context: MCPServerContext, 
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle traditional vault search requests."""
    query = arguments.get("query", "")
    vault_name = arguments.get("vault")
    search_content = arguments.get("search_content", False)
    limit = arguments.get("limit", 20)
    
    if not query:
        return [types.TextContent(type="text", text="Error: Query parameter is required")]
    
    # Determine which vault to use
    if vault_name and vault_name in context.vault_readers:
        vault_reader = context.vault_readers[vault_name]
        search_vault_name = vault_name
    elif vault_name:
        return [types.TextContent(type="text", text=f"Error: Unknown vault '{vault_name}'")]
    else:
        # Use first available vault
        if not context.vault_readers:
            return [types.TextContent(type="text", text="Error: No vaults available")]
        vault_reader = next(iter(context.vault_readers.values()))
        search_vault_name = next(iter(context.vault_readers.keys()))
    
    try:
        results = vault_reader.search_vault(query, search_content, limit)
        
        if not results:
            search_type = "content and filenames" if search_content else "filenames"
            return [types.TextContent(
                type="text", 
                text=f"No results found in {search_type} for query: '{query}'"
            )]
        
        # Format results
        response_lines = [f"Found {len(results)} results in vault '{search_vault_name}' for '{query}':\n"]
        
        # Rank results before formatting
        ranked_results = context.ranker.rank_results(results)

        for i, result in enumerate(ranked_results, 1):
            match_type = result.get('match_type', 'name')
            size_info = f" ({result['size']} bytes)" if result.get('size') else ""
            
            response_lines.append(f"{i}. **{result['path']}** ({match_type} match){size_info}")
            
            # Add content preview if available
            if result.get('content_preview'):
                preview = result['content_preview'].replace('\n', ' ')[:100] + "..."
                response_lines.append(f"   > {preview}")
        
        return [types.TextContent(type="text", text="\n".join(response_lines))]
        
    except ServiceError as e:
        logger.error(f"Error searching vault: {e}")
        return [types.TextContent(type="text", text=f"Error searching vault: {str(e)}")]


async def _handle_search_graph(
    context: MCPServerContext,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle graph search requests."""
    query_note_path = arguments.get("query_note_path", "")
    depth = arguments.get("depth", 1)

    if not query_note_path:
        return [types.TextContent(type="text", text="Error: query_note_path parameter is required")]

    # Check if graph database is available
    if not context.graph_database or not context.graph_database.is_healthy:
        logger.warning("Graph search unavailable, falling back to semantic search.")
        fallback_args = {
            "query": query_note_path,  # Use the note path as the semantic query
            "limit": 10
        }
        fallback_results = await _handle_semantic_search(context, fallback_args)
        
        # Prepend a message to the user
        fallback_message = types.TextContent(
            type="text",
            text="**Graph search is unavailable. Falling back to semantic search.**\n\n"
        )
        return [fallback_message] + fallback_results

    try:
        graph = context.graph_database.get_note_graph(query_note_path, depth)

        if not graph or not graph.get('nodes'):
            return [types.TextContent(type="text", text=f"No results found for query: '{query_note_path}'")]

        # Format enhanced graph output
        response_lines = []
        
        # Header with summary
        response_lines.append(f"# Knowledge Graph for '{query_note_path}'")
        response_lines.append(f"**Traversal Depth:** {depth}")
        response_lines.append(f"**Nodes Found:** {len(graph['nodes'])}")
        response_lines.append(f"**Relationships Found:** {len(graph['relationships'])}")
        response_lines.append("")
        
        # Create node lookup for relationship mapping
        node_lookup = {node['id']: node for node in graph['nodes']}
        
        # Group nodes by whether they are the center node
        center_nodes = [node for node in graph['nodes'] if node.get('center', False)]
        related_nodes = [node for node in graph['nodes'] if not node.get('center', False)]
        
        # Display center node(s)
        if center_nodes:
            response_lines.append("## 🎯 Center Node")
            for node in center_nodes:
                response_lines.append(f"**{node['label']}** (`{node['path']}`)")
                if node.get('tags'):
                    response_lines.append(f"  Tags: {', '.join(node['tags'])}")
                response_lines.append("")
        
        # Display related nodes
        if related_nodes:
            response_lines.append("## 🔗 Connected Notes")
            for i, node in enumerate(related_nodes, 1):
                response_lines.append(f"{i}. **{node['label']}** (`{node['path']}`)")
                if node.get('tags'):
                    response_lines.append(f"   Tags: {', '.join(node['tags'])}")
        
        response_lines.append("")
        
        # Display relationships with enhanced details
        if graph['relationships']:
            response_lines.append("## 🌐 Relationships")
            
            # Group relationships by type
            relationships_by_type = {}
            for rel in graph['relationships']:
                rel_type = rel.get('original_type', rel.get('type', 'UNKNOWN'))
                if rel_type not in relationships_by_type:
                    relationships_by_type[rel_type] = []
                relationships_by_type[rel_type].append(rel)
            
            for rel_type, rels in relationships_by_type.items():
                response_lines.append(f"### {rel_type}")
                for rel in rels:
                    source_node = node_lookup.get(rel['source'])
                    target_node = node_lookup.get(rel['target'])
                    
                    if source_node and target_node:
                        source_label = source_node['label']
                        target_label = target_node['label']
                        response_lines.append(f"- **{source_label}** → **{target_label}**")
                    else:
                        response_lines.append(f"- {rel['source']} → {rel['target']}")
                response_lines.append("")
        
        # Add graph structure summary
        response_lines.append("## 📊 Graph Structure")
        response_lines.append(f"- **Total Connections:** {len(graph['relationships'])}")
        response_lines.append(f"- **Unique Relationship Types:** {len(set(rel.get('original_type', rel.get('type', 'UNKNOWN')) for rel in graph['relationships']))}")
        
        # Show relationship type distribution
        type_counts = {}
        for rel in graph['relationships']:
            rel_type = rel.get('original_type', rel.get('type', 'UNKNOWN'))
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        if type_counts:
            response_lines.append("- **Relationship Distribution:**")
            for rel_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                response_lines.append(f"  - {rel_type}: {count}")

        return [types.TextContent(type="text", text="\n".join(response_lines))]

    except Neo4jError as e:
        logger.error(f"Neo4j error during graph search: {e}")
        raise ServiceError(f"Graph search failed due to database connection issues: {e}") from e
    except ServiceError as e:
        logger.error(f"Graph search error: {e}")
        raise ServiceError(f"Graph search failed: {e}") from e


async def _handle_search_combined(
    context: MCPServerContext,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle combined semantic and keyword search requests."""
    import time
    start_time = time.time()
    
    print("🔄 [ACTUAL MCP HANDLER] Starting combined search execution")
    logger.info("🔄 [ACTUAL MCP HANDLER] Starting combined search execution")
    logger.debug(f"📝 Raw arguments received: {arguments}")
    logger.debug(f"🏷️ Context type: {type(context).__name__}")
    
    # Parse and validate arguments
    query = arguments.get("query", "")
    limit = arguments.get("limit", 10)
    vault_name = arguments.get("vault")
    search_content = arguments.get("search_content", True)
    
    logger.info(f"📋 Parsed arguments - query: '{query[:50]}{'...' if len(query) > 50 else ''}', limit: {limit}, vault_name: {vault_name}, search_content: {search_content}")

    if not query:
        print("❌ Query validation failed: empty query provided")
        logger.warning("❌ Query validation failed: empty query provided")
        return [types.TextContent(type="text", text="Error: Query parameter is required")]

    try:
        combined_results = {}

        # Check context services availability
        logger.debug("🔧 Checking context services availability")
        searcher_available = hasattr(context, 'searcher') and context.searcher is not None
        vault_readers_available = hasattr(context, 'vault_readers') and context.vault_readers
        ranker_available = hasattr(context, 'ranker') and context.ranker is not None
        
        logger.debug(f"🔍 Service availability - searcher: {searcher_available}, vault_readers: {vault_readers_available}, ranker: {ranker_available}")
        
        if not searcher_available:
            logger.error("❌ Searcher service not available in context")
            return [types.TextContent(type="text", text="Error: Semantic search service not available")]
        
        if not vault_readers_available:
            logger.error("❌ Vault readers not available in context")
            return [types.TextContent(type="text", text="Error: Vault reader service not available")]

        # Perform semantic search
        logger.info("🧠 Starting semantic search")
        semantic_start = time.time()
        try:
            semantic_args = {"query": query, "top_k": limit, "vault_name": vault_name}
            logger.debug(f"🧠 Semantic search args: {semantic_args}")
            
            semantic_raw_results = context.searcher.search(**semantic_args)
            semantic_duration = time.time() - semantic_start
            
            logger.info(f"✅ Semantic search completed in {semantic_duration:.3f}s - found {len(semantic_raw_results) if semantic_raw_results else 0} results")
            logger.debug(f"🔍 Semantic results preview: {[{'path': r.path, 'score': f'{r.similarity_score:.3f}'} for r in (semantic_raw_results[:3] if semantic_raw_results else [])]}")
            
            for res in semantic_raw_results:
                combined_results[res.path] = {"type": "semantic", "score": res.similarity_score, "data": res}
        except Exception as e:
            semantic_duration = time.time() - semantic_start
            logger.error(f"❌ Semantic search failed after {semantic_duration:.3f}s: {e}")
            semantic_raw_results = []
            logger.warning("⚠️ Continuing with empty semantic results")

        # Perform keyword search
        logger.info("📝 Starting keyword search")
        keyword_start = time.time()
        try:
            # Select vault reader
            if vault_name and vault_name in context.vault_readers:
                selected_vault = vault_name
                vault_reader = context.vault_readers[vault_name]
                logger.debug(f"📂 Using specified vault: {vault_name}")
            else:
                selected_vault = next(iter(context.vault_readers.keys()))
                vault_reader = context.vault_readers[selected_vault]
                logger.debug(f"📂 Using default vault: {selected_vault}")
            
            keyword_args = {"query": query, "limit": limit, "search_content": search_content}
            logger.debug(f"📝 Keyword search args: {keyword_args}")
            
            keyword_raw_results = vault_reader.search_vault(**keyword_args)
            keyword_duration = time.time() - keyword_start
            
            logger.info(f"✅ Keyword search completed in {keyword_duration:.3f}s - found {len(keyword_raw_results) if keyword_raw_results else 0} results")
            logger.debug(f"📝 Keyword results preview: {[{'path': r.get('path', 'Unknown'), 'type': r.get('match_type', 'unknown')} for r in (keyword_raw_results[:3] if keyword_raw_results else [])]}")
            
            for res in keyword_raw_results:
                if res["path"] not in combined_results:
                    combined_results[res["path"]] = {"type": "keyword", "score": 0.0, "data": res}
                else:
                    # If already found by semantic, enhance with keyword data
                    combined_results[res["path"]]["data"].update(res)
        except Exception as e:
            keyword_duration = time.time() - keyword_start
            logger.error(f"❌ Keyword search failed after {keyword_duration:.3f}s: {e}")
            keyword_raw_results = []
            logger.warning("⚠️ Continuing with empty keyword results")

        # Sort results (semantic first, then keyword)
        logger.info("📊 Ranking and sorting results")
        ranking_start = time.time()
        try:
            if ranker_available:
                sorted_results = context.ranker.merge_and_rank(semantic_raw_results, keyword_raw_results)
                logger.debug(f"✅ Results ranked successfully: {len(sorted_results)} total")
            else:
                logger.warning("⚠️ Ranker not available, using simple concatenation")
                sorted_results = list(semantic_raw_results) + list(keyword_raw_results)
            
            ranking_duration = time.time() - ranking_start
            logger.debug(f"📊 Ranking completed in {ranking_duration:.3f}s")
        except Exception as e:
            ranking_duration = time.time() - ranking_start
            logger.error(f"❌ Ranking failed after {ranking_duration:.3f}s: {e}")
            sorted_results = list(semantic_raw_results) + list(keyword_raw_results)

        if not sorted_results:
            logger.warning("❌ No results found in combined search")
            return [types.TextContent(type="text", text=f"No results found for query: '{query}'")]

        # Format response
        logger.info("📋 Formatting response")
        response_lines = [f"Found {len(sorted_results)} combined results for '{query}':\n"]
        results_displayed = 0
        
        for i, item in enumerate(sorted_results[:limit], 1):
            data = item
            if isinstance(item, SearchResult):
                response_lines.append(f"{i}. **[SEMANTIC] {data.path}** (vault: {data.vault_name}, score: {data.similarity_score:.3f})")
                results_displayed += 1
            else:
                match_type = data.get("match_type", "name")
                size_info = f" ({data['size']} bytes)" if data.get("size") else ""
                response_lines.append(f"{i}. **[KEYWORD] {data['path']}** ({match_type} match){size_info}")
                if data.get("content_preview"):
                    preview = data['content_preview'].replace('\n', ' ')[:100] + "..."
                    response_lines.append(f"   > {preview}")
                results_displayed += 1
        
        total_duration = time.time() - start_time
        logger.info(f"🎉 Combined search completed successfully in {total_duration:.3f}s")
        logger.info(f"📈 Results summary - displayed: {results_displayed}, total found: {len(sorted_results)}, response lines: {len(response_lines)}")
        
        return [types.TextContent(type="text", text="\n".join(response_lines))]
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"💥 Combined search error after {total_duration:.3f}s: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        logger.error(f"📋 Error details: {str(e)}")
        return [types.TextContent(type="text", text=f"Error in combined search: {str(e)}")]


async def _handle_get_health_status(
    context: MCPServerContext,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle health status requests."""
    try:
        health_status = context.health_checker.get_overall_health()
        return [types.TextContent(type="text", text=json.dumps(health_status, indent=2))]
    except ServiceError as e:
        logger.error(f"Error getting health status: {e}")
        return [types.TextContent(type="text", text=f"Error getting health status: {str(e)}")]


async def _handle_get_performance_metrics(
    context: MCPServerContext,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """Handle performance metrics requests."""
    if not context.metrics:
        return [types.TextContent(
            type="text", 
            text="Performance metrics are disabled. Set JARVIS_METRICS_ENABLED=true to enable."
        )]
    
    try:
        reset_after_read = arguments.get("reset_after_read", False)
        filter_prefix = arguments.get("filter_prefix")
        
        # Get all metrics
        metrics_data = context.metrics.get_metrics()
        
        # Filter metrics if prefix specified
        if filter_prefix:
            filtered_metrics = {}
            for metric_name, metric_data in metrics_data["metrics"].items():
                if metric_name.startswith(filter_prefix):
                    filtered_metrics[metric_name] = metric_data
            metrics_data["metrics"] = filtered_metrics
        
        # Add cache statistics if available
        if context.mcp_cache:
            cache_stats = context.mcp_cache.get_stats()
            metrics_data["cache_stats"] = cache_stats
        
        # Format as JSON with readable structure
        response = json.dumps(metrics_data, indent=2, default=str)
        
        # Reset metrics if requested
        if reset_after_read:
            context.metrics.reset_metrics()
            response += "\n\n✅ Metrics have been reset."
        
        return [types.TextContent(type="text", text=response)]
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return [types.TextContent(type="text", text=f"Error getting performance metrics: {str(e)}")]


async def run_mcp_server(
    vaults: Dict[str, Path],
    database_path: Path,
    settings: Optional[JarvisSettings] = None,
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
            logger.error("💡   1. Check file permissions: ls -la '{}'".format(database_path.parent))
            logger.error("💡   2. Create directory: mkdir -p '{}'".format(database_path.parent))
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
                    logger.error(f"❌ File watching disabled due to invalid vault path")
                    watch = False
                    break
                elif not vault_path.is_dir():
                    logger.error(f"❌ Vault path is not a directory: {vault_name} -> {vault_path}")
                    logger.error(f"❌ File watching disabled due to invalid vault path")
                    watch = False
                    break
                else:
                    logger.debug(f"✅ Vault path validated: {vault_name} -> {vault_path}")
                    # Log special iCloud handling
                    if "iCloud" in str(vault_path):
                        logger.info(f"📱 iCloud vault detected: {vault_name} -> {vault_path}")
                        logger.info(f"📱 iCloud sync may cause additional file system events")
            
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
                        logger.debug(f"💾 Worker database connection created successfully")
                        
                        # Get worker database info for logging
                        worker_db_info = worker_initializer.get_database_info()
                        logger.debug(f"💾 Worker database ready: {worker_db_info['note_count']} notes, {worker_db_info['size_mb']} MB")
                        
                        # Continue with vector worker setup
                        logger.info("🤖 Initializing vector encoder")
                        encoder = VectorEncoder()
                        logger.debug(f"🧠 Vector encoder initialized")
                        
                        logger.info("👷 Creating vector worker")
                        # Create and start the vector worker
                        vector_worker = VectorWorker(
                            database=worker_database,
                            encoder=encoder,
                            vaults=vaults,
                            enable_watching=True,
                            auto_index=False  # Don't auto-index on startup
                        )
                        logger.debug(f"👷 Vector worker created successfully")
                        
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
            logger.error(f"📋 Full exception details: {str(e)}")
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
    import sys
    import os
    
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
    
    # Run the server
    asyncio.run(run_mcp_server(default_vaults, default_db_path))


if __name__ == "__main__":
    main()