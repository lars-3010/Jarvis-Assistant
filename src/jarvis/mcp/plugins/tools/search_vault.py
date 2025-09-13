"""
Search Vault Plugin for MCP Tools.

This plugin provides traditional text-based search capabilities
for finding files by filename or content in vaults.
Supports structured JSON output via `format: "json"`.
"""

import json
import time
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IVaultReader
from jarvis.mcp.plugins.base import VaultPlugin
from jarvis.mcp.structured import vault_search_to_json
from jarvis.services.search import ResultRanker
from jarvis.utils.errors import PluginError, ServiceError

# Lazy import to avoid circular dependencies
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class SearchVaultPlugin(VaultPlugin):
    """Plugin for traditional text-based vault search."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "search-vault"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Search for files in vault by filename or content (traditional search)"

    @property
    def version(self) -> str:
        """Get the plugin version."""
        return "1.0.0"

    @property
    def author(self) -> str:
        """Get the plugin author."""
        return "Jarvis Assistant"

    @property
    def tags(self) -> list[str]:
        """Get plugin tags."""
        return ["vault", "search", "text", "files"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVaultReader]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import SearchSchemaConfig, create_search_schema

        # Create standardized search schema with custom configuration
        schema_config = SearchSchemaConfig(
            query_required=True,
            enable_similarity_threshold=False,  # Not applicable for text search
            enable_vault_selection=True,
            enable_content_search=True,  # Key feature for vault search
            max_limit=100,  # Higher limit for text search
            default_limit=20,
            supported_formats=["json"]  # Always returns JSON for structured data
        )

        # Generate standardized schema
        input_schema = create_search_schema(schema_config)

        # Customize query description for vault search
        input_schema["properties"]["query"]["description"] = "Search term for filenames or content"

        # Keep `format` field but restrict to JSON for uniformity
        if "format" in input_schema["properties"]:
            input_schema["properties"]["format"].update({
                "enum": ["json"],
                "default": "json",
            })

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute vault search."""
        query = arguments.get("query", "").strip()
        vault_name = arguments.get("vault")
        search_content = arguments.get("search_content", False)
        limit = arguments.get("limit", 20)
        start_time = time.time()

        # Validate input
        if not query:
            return [types.TextContent(
                type="text",
                text="Error: Query parameter is required and cannot be empty"
            )]

        if limit < 1 or limit > 100:
            return [types.TextContent(
                type="text",
                text="Error: Limit must be between 1 and 100"
            )]

        try:
            # Get vault reader service
            if not self.container:
                raise PluginError("Service container not available")

            vault_reader = self.container.get(IVaultReader)
            if not vault_reader:
                raise PluginError("Vault reader service not available")

            # Note: Current interface doesn't support vault selection
            # This is a limitation that could be addressed in future versions
            if vault_name:
                logger.warning(f"Vault selection '{vault_name}' requested but not supported by current interface")

            # Get optional ranker service
            ranker = None
            try:
                ranker = self.container.get(ResultRanker)
            except:
                pass  # Ranker is optional

            # Perform search
            try:
                results = vault_reader.search_vault(query, search_content, limit)
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error searching vault: {e!s}"
                )]

            if not results:
                search_type = "content and filenames" if search_content else "filenames"
                return [types.TextContent(
                    type="text",
                    text=f"No results found in {search_type} for query: '{query}'"
                )]

            # Format results
            response_lines = [f"Found {len(results)} results for '{query}':\n"]

            # Rank results if ranker is available
            final_results = results
            if ranker:
                try:
                    final_results = ranker.rank_results(results)
                except Exception as e:
                    logger.warning(f"Result ranking failed, using original order: {e}")

            for i, result in enumerate(final_results, 1):
                # Handle both dictionary and object results
                if isinstance(result, dict):
                    path = result.get('path', 'Unknown')
                    match_type = result.get('match_type', 'name')
                    size = result.get('size')
                    content_preview = result.get('content_preview')
                else:
                    # Handle SearchResult objects
                    path = getattr(result, 'path', 'Unknown')
                    match_type = getattr(result, 'match_type', 'name')
                    size = getattr(result, 'size', None)
                    content_preview = getattr(result, 'content_preview', None)

                size_info = f" ({size} bytes)" if size else ""
                response_lines.append(f"{i}. **{path}** ({match_type} match){size_info}")

                # Add content preview if available
                if content_preview:
                    preview = content_preview.replace('\n', ' ')[:100]
                    if len(content_preview) > 100:
                        preview += "..."
                    response_lines.append(f"   > {preview}")

            duration_ms = int((time.time() - start_time) * 1000)
            payload = vault_search_to_json(
                query=query,
                results=final_results,
                search_content=search_content,
                limit=limit,
                execution_time_ms=duration_ms,
            )
            payload["correlation_id"] = str(uuid4())
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except ServiceError as e:
            logger.error(f"Error searching vault: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error searching vault: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error searching vault: {e}")
            raise PluginError(f"Vault search failed: {e!s}") from e
