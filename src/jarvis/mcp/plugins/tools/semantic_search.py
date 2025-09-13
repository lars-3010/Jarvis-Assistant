"""
Semantic Search Plugin for MCP Tools.

This plugin provides semantic search capabilities across vault content
using natural language queries and vector embeddings.
"""

import json
import time
from typing import Any

from jarvis.core.interfaces import IMetrics, IVectorSearcher
from jarvis.mcp.plugins.base import SearchPlugin
from jarvis.mcp.structured import semantic_search_to_json
from jarvis.services.search import ResultRanker
from jarvis.utils.errors import PluginError, ServiceError

# Lazy import to avoid circular dependencies
import logging
from mcp import types

logger = logging.getLogger(__name__)


class SemanticSearchPlugin(SearchPlugin):
    """Plugin for semantic search across vault content."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "search-semantic"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Perform semantic search across vault content using natural language queries"

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
        return ["search", "semantic", "embeddings", "ai"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVectorSearcher]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import SearchSchemaConfig, create_search_schema

        # Create standardized search schema with custom configuration
        schema_config = SearchSchemaConfig(
            query_required=True,
            enable_similarity_threshold=True,
            enable_vault_selection=True,
            max_limit=50,
            default_limit=10,
            supported_formats=["json"]
        )

        # Generate standardized schema
        input_schema = create_search_schema(schema_config)

        # Customize query description for semantic search
        input_schema["properties"]["query"]["description"] = "Natural language search query"

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute semantic search."""
        query = arguments.get("query", "").strip()
        limit = arguments.get("limit", 10)
        vault_name = arguments.get("vault")
        similarity_threshold = arguments.get("similarity_threshold")
        output_format = arguments.get("format", "json")

        start_time = time.time()

        # Validate input
        if not query:
            return [types.TextContent(
                type="text",
                text="Error: Query parameter is required and cannot be empty"
            )]

        if limit < 1 or limit > 50:
            return [types.TextContent(
                type="text",
                text="Error: Limit must be between 1 and 50"
            )]

        try:
            # Get required services
            searcher = self.container.get(IVectorSearcher) if self.container else None
            if not searcher:
                raise PluginError("Vector searcher service not available")

            # Get optional services
            metrics = None
            ranker = None
            if self.container:
                try:
                    metrics = self.container.get(IMetrics)
                except:
                    pass  # Metrics are optional

                try:
                    ranker = self.container.get(ResultRanker)
                except:
                    pass  # Ranker is optional

            # Perform semantic search with metrics if available
            if metrics:
                with metrics.time_operation("mcp_tool_search_semantic"):
                    results = searcher.search(
                        query=query,
                        top_k=limit,
                        vault_name=vault_name,
                        similarity_threshold=similarity_threshold
                    )
                metrics.record_counter("mcp_tool_search_semantic_results", len(results))
            else:
                results = searcher.search(
                    query=query,
                    top_k=limit,
                    vault_name=vault_name,
                    similarity_threshold=similarity_threshold
                )

            # Rank results if ranker is available
            final_results = results
            if ranker:
                try:
                    final_results = ranker.rank_results(results)
                except Exception as e:
                    logger.warning(f"Result ranking failed, using original order: {e}")

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Return JSON format if requested
            if output_format == "json":
                json_data = semantic_search_to_json(
                    query=query,
                    results=final_results,
                    execution_time_ms=execution_time_ms,
                    similarity_threshold=similarity_threshold,
                )
                return [types.TextContent(type="text", text=json.dumps(json_data, indent=2))]

            # Default markdown format
            if not final_results:
                return [types.TextContent(
                    type="text",
                    text=f"No results found for query: '{query}'"
                )]

            # Format results
            response_lines = [f"Found {len(final_results)} results for '{query}':\n"]

            for i, result in enumerate(final_results, 1):
                score = f"{result.similarity_score:.3f}"
                response_lines.append(
                    f"{i}. **{result.path}** (vault: {result.vault_name}, score: {score})"
                )

            return [types.TextContent(type="text", text="\n".join(response_lines))]

        except ServiceError as e:
            logger.error(f"Semantic search error: {e}")
            return [types.TextContent(
                type="text",
                text=f"Search error: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error in semantic search: {e}")
            raise PluginError(f"Semantic search failed: {e!s}") from e
