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
        # Standardized schema using search template + additional properties
        from jarvis.mcp.schemas import SearchSchemaConfig, create_search_schema

        schema_config = SearchSchemaConfig(
            query_required=True,
            enable_similarity_threshold=False,  # Not used directly in GraphRAG
            enable_vault_selection=True,
            max_limit=20,
            default_limit=5,
            supported_formats=["json"],  # JSON-only for AI consumption
            additional_properties={
                "mode": {
                    "type": "string",
                    "enum": ["quick", "focused", "comprehensive"],
                    "default": "quick",
                    "description": "Search depth and thoroughness",
                },
                "max_sources": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of semantic sources to expand",
                },
                "depth": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Graph neighborhood depth for expansion",
                },
                "include_content": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include content previews for top sources",
                },
                "semantic_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Optional threshold to filter semantic results (0.0-1.0)",
                },
                "graph_boost": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Weight to emphasize graph connectivity in reranking",
                },
            },
        )

        input_schema = create_search_schema(schema_config)
        input_schema["properties"]["query"]["description"] = "Natural language search query"

        return types.Tool(name=self.name, description=self.description, inputSchema=input_schema)

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
                include_content=bool(arguments.get("include_content", False)),
                semantic_threshold=arguments.get("semantic_threshold"),
                graph_boost=float(arguments.get("graph_boost", 0.3)),
            )

            # Execute comprehensive GraphRAG search
            result = await graphrag_service.search(query_config)

            # Convert to structured response using features.GraphRAGService result
            duration_ms = int((time.time() - start) * 1000)

            # Sources are already in dict form in the features service result
            sources = result.sources if isinstance(result.sources, list) else []

            # Build a single GraphData item aggregating nodes/relationships
            rel_types = set()
            rel_dicts: list[dict] = []
            for rel in result.relationships or []:
                if hasattr(rel, "__dict__"):
                    d = {**rel.__dict__}
                elif isinstance(rel, dict):
                    d = dict(rel)
                else:
                    d = {}
                rel_types.add(d.get("relationship_type", d.get("type", "UNKNOWN")))
                rel_dicts.append(d)

            node_dicts: list[dict] = []
            for node in result.nodes or []:
                if hasattr(node, "__dict__"):
                    node_dicts.append({**node.__dict__})
                elif isinstance(node, dict):
                    node_dicts.append(dict(node))

            metrics_dict = {
                "nodes": len(node_dicts),
                "relationships": len(rel_dicts),
                "relationship_types": len(rel_types),
            }

            center_path = sources[0]["path"] if sources else query
            graphs = [
                GraphData(
                    center_path=center_path,
                    nodes=node_dicts,
                    relationships=rel_dicts,
                    metrics=metrics_dict,
                )
            ]

            limits = {
                "mode": query_config.mode,
                "max_sources": query_config.max_sources,
                "depth": query_config.depth,
                "include_content": query_config.include_content,
                "semantic_threshold": query_config.semantic_threshold,
                "graph_boost": query_config.graph_boost,
            }

            payload = graphrag_to_json(query, sources, graphs, duration_ms, limits)
            payload["correlation_id"] = str(uuid4())
            payload["graph_metrics"] = result.graph_metrics if isinstance(result.graph_metrics, dict) else {}
            payload["search_path"] = result.search_path

            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except ServiceError as e:
            logger.error(f"GraphRAG service error: {e}")
            return [types.TextContent(type="text", text=json.dumps({"error": f"GraphRAG service failed: {e!s}"}))]
        except Exception as e:
            logger.error(f"Unexpected error in GraphRAG search: {e}")
            return [types.TextContent(type="text", text=json.dumps({"error": f"Search failed: {e!s}"}))]
