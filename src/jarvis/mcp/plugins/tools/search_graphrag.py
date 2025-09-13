"""
GraphRAG Search Plugin for MCP Tools.

This plugin provides comprehensive GraphRAG (Graph-Retrieval-Augmented Generation) search
that combines semantic retrieval with knowledge graph traversal for enhanced search results.
"""

import json
import time
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IGraphDatabase, IMetrics, IVaultReader, IVectorSearcher
from jarvis.mcp.plugins.base import SearchPlugin
from jarvis.mcp.structured import GraphData, graphrag_to_json
from jarvis.services.graphrag import GraphRAGQuery, GraphRAGService
from jarvis.utils.errors import PluginError, ServiceError
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class SearchGraphRAGPlugin(SearchPlugin):
    """Comprehensive GraphRAG search tool."""

    @property
    def name(self) -> str:
        return "search-graphrag"

    @property
    def description(self) -> str:
        return "Comprehensive GraphRAG search combining semantic retrieval with knowledge graph traversal"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def author(self) -> str:
        return "Jarvis Assistant"

    @property
    def tags(self) -> list[str]:
        return ["search", "graph", "graphrag", "ai", "semantic", "knowledge-graph"]

    def get_required_services(self) -> list[type]:
        return [IVectorSearcher, IGraphDatabase]

    def get_tool_definition(self) -> types.Tool:
        from jarvis.mcp.schemas import get_schema_manager
        schema_manager = get_schema_manager()
        
# Use advanced search schema for GraphRAG
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "mode": {
                    "type": "string", 
                    "enum": ["quick", "focused", "comprehensive"], 
                    "default": "quick",
                    "description": "Search depth and thoroughness"
                },
                "max_sources": {
                    "type": "integer", 
                    "default": 5, 
                    "minimum": 1, 
                    "maximum": 20,
                    "description": "Maximum number of source documents to analyze"
                },
                "depth": {
                    "type": "integer", 
                    "default": 1, 
                    "minimum": 1, 
                    "maximum": 3,
                    "description": "Graph traversal depth"
                },
                "vault": {
                    "type": "string",
                    "description": "Optional vault name to search within"
                },
                "include_content": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include content previews"
                },
                "enable_clustering": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to enable result clustering"
                }
            },
            "required": ["query"]
        }

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute comprehensive GraphRAG search using the new service architecture."""
        query = arguments.get("query", "").strip()
        if not query:
            return [types.TextContent(type="text", text=json.dumps({"error": "query is required"}))]

        # Get services
        if not self.container:
            raise PluginError("Service container not available")

        searcher = self.container.get(IVectorSearcher)
        graph_db = self.container.get(IGraphDatabase)
        vault_reader = self.container.get(IVaultReader)
        metrics = self.container.get(IMetrics)

        if not searcher:
            raise PluginError("Vector searcher not available")
        if not graph_db or not graph_db.is_healthy:
            return [types.TextContent(type="text", text=json.dumps({"error": "graph database unavailable"}))]

        start = time.time()

        try:
            # Initialize GraphRAG service
            graphrag_service = GraphRAGService(
                vector_searcher=searcher,
                graph_db=graph_db,
                vault_reader=vault_reader,
                metrics=metrics
            )

            # Build query configuration
            query_config = GraphRAGQuery(
                query=query,
                mode=arguments.get("mode", "quick"),
                max_sources=int(arguments.get("max_sources", 5)),
                depth=int(arguments.get("depth", 1)),
                vault_name=arguments.get("vault"),
                include_content=arguments.get("include_content", True),
                enable_clustering=arguments.get("enable_clustering", True)
            )

            # Execute comprehensive GraphRAG search
            result = await graphrag_service.search(query_config)

            # Convert to structured response format
            sources = [
                {
                    "path": str(item.path),
                    "vault_name": item.vault_name,
                    "semantic_score": float(item.semantic_score),
                    "graph_score": float(item.graph_score),
                    "unified_score": float(item.unified_score),
                    "content_preview": item.content_preview[:200] + "..." if item.content_preview and len(item.content_preview) > 200 else item.content_preview,
                    "cluster_id": item.cluster_id
                }
                for item in result.ranked_results
            ]

            graph_items = []
            for graph_data in result.graph_data:
                nodes = [asdict(node) for node in graph_data.nodes]
                relationships = [asdict(rel) for rel in graph_data.relationships]

                metrics_dict = {
                    "nodes": len(nodes),
                    "relationships": len(relationships),
                    "relationship_types": len({rel.get("type", "UNKNOWN") for rel in relationships}),
                    "cluster_count": len(result.clusters) if result.clusters else 0
                }

                graph_items.append(GraphData(
                    center_path=graph_data.center_path,
                    nodes=nodes,
                    relationships=relationships,
                    metrics=metrics_dict
                ))

            # Add cluster information
            clusters = [
                {
                    "id": cluster.id,
                    "center_path": cluster.center_path,
                    "member_paths": cluster.member_paths,
                    "strength": cluster.strength,
                    "description": cluster.description
                }
                for cluster in result.clusters
            ] if result.clusters else []

            duration_ms = int((time.time() - start) * 1000)
            limits = {
                "mode": query_config.mode,
                "max_sources": query_config.max_sources,
                "depth": query_config.depth,
                "include_content": query_config.include_content,
                "enable_clustering": query_config.enable_clustering
            }

            # Build comprehensive payload
            payload = graphrag_to_json(query, sources, graph_items, duration_ms, limits)
            payload.update({
                "correlation_id": str(uuid4()),
                "clusters": clusters,
                "search_metadata": {
                    "total_semantic_results": len(result.semantic_results),
                    "total_graph_expansions": len(result.graph_data),
                    "reranking_applied": True,
                    "clustering_applied": query_config.enable_clustering and len(clusters) > 0
                }
            })

            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except ServiceError as e:
            logger.error(f"GraphRAG service error: {e}")
            return [types.TextContent(type="text", text=json.dumps({"error": f"GraphRAG service failed: {e!s}"}))]
        except Exception as e:
            logger.error(f"Unexpected error in GraphRAG search: {e}")
            return [types.TextContent(type="text", text=json.dumps({"error": f"Search failed: {e!s}"}))]

