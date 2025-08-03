"""
Semantic Search Plugin for MCP Tools.

This plugin provides semantic search capabilities across vault content
using natural language queries and vector embeddings.
"""

from typing import Dict, Any, List, Type

from mcp import types
from jarvis.mcp.plugins.base import SearchPlugin
from jarvis.core.interfaces import IVectorSearcher, IMetrics
from jarvis.services.ranking import ResultRanker
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import PluginError, ServiceError

logger = setup_logging(__name__)


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
    def tags(self) -> List[str]:
        """Get plugin tags."""
        return ["search", "semantic", "embeddings", "ai"]
    
    def get_required_services(self) -> List[Type]:
        """Get required service interfaces."""
        return [IVectorSearcher]
    
    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "vault": {
                        "type": "string",
                        "description": "Optional vault name to search within"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["query"]
            }
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute semantic search."""
        query = arguments.get("query", "").strip()
        limit = arguments.get("limit", 10)
        vault_name = arguments.get("vault")
        similarity_threshold = arguments.get("similarity_threshold")
        
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
            
            if not results:
                return [types.TextContent(
                    type="text", 
                    text=f"No results found for query: '{query}'"
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
                score = f"{result.similarity_score:.3f}"
                response_lines.append(
                    f"{i}. **{result.path}** (vault: {result.vault_name}, score: {score})"
                )
            
            return [types.TextContent(type="text", text="\n".join(response_lines))]
            
        except ServiceError as e:
            logger.error(f"Semantic search error: {e}")
            return [types.TextContent(
                type="text", 
                text=f"Search error: {str(e)}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error in semantic search: {e}")
            raise PluginError(f"Semantic search failed: {str(e)}") from e