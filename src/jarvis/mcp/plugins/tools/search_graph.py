"""
Search Graph Plugin for MCP Tools.

This plugin provides graph-based search capabilities for exploring
relationships between notes in the knowledge graph.
"""

import json
import time
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IGraphDatabase, IVaultReader, IVectorSearcher
from jarvis.mcp.plugins.base import GraphPlugin
from jarvis.mcp.structured import graph_search_to_json, semantic_fallback_to_json

# Lazy import to avoid circular dependencies
from jarvis.utils.errors import PluginError, ServiceError
import logging
from mcp import types

logger = logging.getLogger(__name__)

# Constants
MAX_DEPTH = 5
MIN_DEPTH = 1
MAX_DISCOVERED_NOTES = 3
SEARCH_LIMIT = 5
FALLBACK_LIMIT = 10


class SearchGraphPlugin(GraphPlugin):
    """Plugin for graph-based knowledge search."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "search-graph"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Search for notes and their relationships in the knowledge graph"

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
        return ["graph", "relationships", "knowledge", "network"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IGraphDatabase]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition with standardized schema."""
        # Lazy import to avoid circular dependencies
        from jarvis.mcp.schemas import create_graph_schema

        # Create standardized graph schema
        input_schema = create_graph_schema(
            enable_depth_control=True,
            max_depth=MAX_DEPTH
        )

        # Customize query description for this specific tool
        input_schema["properties"]["query_note_path"]["description"] = (
            "The path to the note to use as the center of the search, or keywords to search for relevant notes"
        )

        # Keep `format` field but restrict to JSON for uniformity
        if "format" in input_schema["properties"]:
            input_schema["properties"]["format"] = {
                "type": "string",
                "enum": ["json"],
                "default": "json",
                "description": "Response format"
            }

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema
        )

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute graph search with enhanced keyword search capability."""
        query_note_path = arguments.get("query_note_path", "").strip()
        depth = arguments.get("depth", 1)
        start_time = time.time()

        # Validate input
        validation_error = self._validate_input(query_note_path, depth)
        if validation_error:
            return validation_error

        try:
            # Get services
            if not self.container:
                raise PluginError("Service container not available")

            graph_database = self.container.get(IGraphDatabase)

            # Check if graph database is available and healthy
            if not graph_database or not graph_database.is_healthy:
                logger.warning("Graph search unavailable, falling back to semantic search")
                payload = await self._fallback_to_semantic_search_structured(query_note_path, start_time)
                return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

            logger.info(f"Graph search: Starting search for query_note_path: '{query_note_path}' with depth: {depth}")

            # Try exact path first, then keyword search
            payload = await self._perform_graph_search_structured(graph_database, query_note_path, depth, start_time)
            payload["correlation_id"] = str(uuid4())
            return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except ServiceError as e:
            logger.error(f"Error in graph search: {e}")
            return [types.TextContent(
                type="text",
                text=f"Error in graph search: {e!s}"
            )]
        except Exception as e:
            logger.error(f"Unexpected error in graph search: {e}")
            raise PluginError(f"Graph search failed: {e!s}") from e

    async def _perform_graph_search_structured(
        self, graph_database, query_note_path: str, depth: int, start_time: float
    ) -> dict:
        """Structured variant of graph search for JSON output."""
        logger.info(f"🔍 [JSON] Starting graph search for: '{query_note_path}' (depth: {depth})")

        # Phase 1: exact path
        try:
            graph = graph_database.get_note_graph(query_note_path, depth)
            if graph and graph.get("nodes"):
                duration_ms = int((time.time() - start_time) * 1000)
                return graph_search_to_json(
                    query=query_note_path,
                    depth=depth,
                    graphs=[(query_note_path, graph)],
                    mode="exact",
                    execution_time_ms=duration_ms,
                )
        except Exception as e:
            logger.info(f"⚠️ [JSON] Exact path search failed: {e}")

        # Phase 2: keyword discovery
        note_paths = await self._find_notes_by_keywords(query_note_path)
        if not note_paths:
            duration_ms = int((time.time() - start_time) * 1000)
            return graph_search_to_json(
                query=query_note_path,
                depth=depth,
                graphs=[],
                mode="keyword_fallback",
                execution_time_ms=duration_ms,
                discovered_notes=[],
            )

        all_graphs: list[tuple[str, dict]] = []
        for note_path in note_paths[:MAX_DISCOVERED_NOTES]:
            try:
                g = graph_database.get_note_graph(note_path, depth)
                if g and g.get("nodes"):
                    all_graphs.append((note_path, g))
            except Exception as e:
                logger.warning(f"⚠️ [JSON] Failed to build graph for '{note_path}': {e}")

        duration_ms = int((time.time() - start_time) * 1000)
        return graph_search_to_json(
            query=query_note_path,
            depth=depth,
            graphs=all_graphs,
            mode="keyword_fallback",
            execution_time_ms=duration_ms,
            discovered_notes=note_paths,
        )

    async def _fallback_to_semantic_search_structured(self, query_note_path: str, start_time: float) -> dict:
        """Structured JSON for semantic fallback when graph unavailable."""
        try:
            searcher = self.container.get(IVectorSearcher) if self.container else None
            results = []
            if searcher:
                results = searcher.search(query=query_note_path, top_k=FALLBACK_LIMIT)
            duration_ms = int((time.time() - start_time) * 1000)
            return semantic_fallback_to_json(query_note_path, results, duration_ms)
        except Exception as e:
            logger.error(f"Semantic fallback structured failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                "query": query_note_path,
                "mode": "fallback_semantic",
                "execution_time_ms": duration_ms,
                "error": str(e),
                "results": [],
            }

    def _validate_input(self, query_note_path: str, depth: int) -> list[types.TextContent] | None:
        """Validate input parameters."""
        if not query_note_path:
            return [types.TextContent(
                type="text",
                text="Error: query_note_path parameter is required and cannot be empty"
            )]

        if depth < MIN_DEPTH or depth > MAX_DEPTH:
            return [types.TextContent(
                type="text",
                text=f"Error: Depth must be between {MIN_DEPTH} and {MAX_DEPTH}"
            )]

        return None

    async def _perform_graph_search(self, graph_database, query_note_path: str, depth: int) -> list[types.TextContent]:
        """Perform graph search with exact path and keyword fallback."""
        logger.info(f"🔍 Starting graph search for: '{query_note_path}' (depth: {depth})")

        # Try exact path first (for backward compatibility)
        try:
            logger.info(f"📍 Phase 1: Attempting exact path search for: '{query_note_path}'")
            graph = graph_database.get_note_graph(query_note_path, depth)

            if graph and graph.get('nodes'):
                logger.info(f"✅ Exact path match found! Nodes: {len(graph['nodes'])}, Relationships: {len(graph['relationships'])}")
                return [types.TextContent(type="text", text=self._format_graph_output(query_note_path, graph, depth, "exact path"))]

            logger.info("❌ No exact path match found, proceeding to keyword search")

        except Exception as e:
            logger.info(f"⚠️ Exact path search failed: {e}, proceeding to keyword search")

        # Try keyword search to find matching notes
        logger.info(f"🔍 Phase 2: Starting keyword search for: '{query_note_path}'")
        note_paths = await self._find_notes_by_keywords(query_note_path)

        if not note_paths:
            logger.warning(f"❌ No notes found matching query: '{query_note_path}'")
            return [types.TextContent(
                type="text",
                text=f"No notes found matching query: '{query_note_path}'"
            )]

        logger.info(f"📋 Found {len(note_paths)} matching notes: {note_paths}")

        # Build graphs around discovered notes
        logger.info(f"🌐 Phase 3: Building graphs around top {min(len(note_paths), MAX_DISCOVERED_NOTES)} discovered notes")
        all_graphs = []
        for i, note_path in enumerate(note_paths[:MAX_DISCOVERED_NOTES], 1):
            try:
                logger.info(f"📊 Building graph {i}/{min(len(note_paths), MAX_DISCOVERED_NOTES)}: '{note_path}'")
                graph = graph_database.get_note_graph(note_path, depth)
                if graph and graph.get('nodes'):
                    logger.info(f"✅ Graph {i} built successfully: {len(graph['nodes'])} nodes, {len(graph['relationships'])} relationships")
                    all_graphs.append((note_path, graph))
                else:
                    logger.info(f"❌ Graph {i} is empty or invalid")
            except Exception as e:
                logger.warning(f"⚠️ Failed to build graph {i} for '{note_path}': {e}")

        if not all_graphs:
            logger.warning(f"❌ No valid graphs found from {len(note_paths)} discovered notes")
            return [types.TextContent(
                type="text",
                text=f"Found {len(note_paths)} matching notes, but no graph relationships were found"
            )]

        logger.info(f"🎉 Successfully built {len(all_graphs)} graphs from {len(note_paths)} discovered notes")
        # Format combined results
        return [types.TextContent(type="text", text=self._format_keyword_search_results(query_note_path, all_graphs, depth, note_paths))]

    async def _find_notes_by_keywords(self, query: str) -> list[str]:
        """Find notes matching keywords using semantic and keyword search."""
        logger.info(f"🔍 Keyword search phase started for query: '{query}'")

        searcher = self.container.get(IVectorSearcher) if self.container else None
        vault_reader = self.container.get(IVaultReader) if self.container else None

        logger.info(f"📋 Available services: semantic_search={searcher is not None}, vault_reader={vault_reader is not None}")

        if not searcher and not vault_reader:
            logger.warning("❌ No search services available for keyword search")
            return []

        note_paths = []

        # Try semantic search
        logger.info("🧠 Attempting semantic search...")
        semantic_paths = self._try_semantic_search(searcher, query)
        note_paths.extend(semantic_paths)
        logger.info(f"🧠 Semantic search returned {len(semantic_paths)} paths")

        # Try keyword search
        logger.info("📝 Attempting keyword search...")
        keyword_paths = self._try_keyword_search(vault_reader, query, note_paths)
        note_paths.extend(keyword_paths)
        logger.info(f"📝 Keyword search returned {len(keyword_paths)} new paths")

        # Remove duplicates while preserving order
        unique_paths = list(dict.fromkeys(note_paths))
        logger.info(f"✅ Keyword search completed: {len(unique_paths)} unique notes found")
        logger.info(f"📋 Discovered notes: {unique_paths}")
        return unique_paths

    def _try_semantic_search(self, searcher, query: str) -> list[str]:
        """Try semantic search and return found paths."""
        try:
            if searcher:
                logger.info(f"🧠 Executing semantic search with query: '{query}' (top_k={SEARCH_LIMIT})")
                semantic_results = searcher.search(query=query, top_k=SEARCH_LIMIT)
                if semantic_results:
                    paths = [result.path for result in semantic_results]
                    scores = [f"{result.similarity_score:.3f}" for result in semantic_results]
                    logger.info(f"🧠 Semantic search found {len(semantic_results)} notes:")
                    for i, (path, score) in enumerate(zip(paths, scores, strict=False), 1):
                        logger.info(f"  {i}. {path} (score: {score})")
                    return paths
                else:
                    logger.info("🧠 Semantic search returned no results")
            else:
                logger.info("🧠 Semantic search service not available")
        except Exception as e:
            logger.warning(f"🧠 Semantic search failed: {e}")
        return []

    def _try_keyword_search(self, vault_reader, query: str, existing_paths: list[str]) -> list[str]:
        """Try keyword search and return new paths not in existing_paths."""
        try:
            if vault_reader:
                logger.info(f"📝 Executing keyword search with query: '{query}' (limit={SEARCH_LIMIT}, search_content=True)")
                keyword_results = vault_reader.search_vault(query=query, search_content=True, limit=SEARCH_LIMIT)
                if keyword_results:
                    new_paths = []
                    logger.info(f"📝 Keyword search found {len(keyword_results)} results:")
                    for i, result in enumerate(keyword_results, 1):
                        path = result.get('path') if isinstance(result, dict) else getattr(result, 'path', None)
                        match_type = result.get('match_type', 'unknown') if isinstance(result, dict) else getattr(result, 'match_type', 'unknown')
                        is_duplicate = path in existing_paths
                        status = "duplicate" if is_duplicate else "new"
                        logger.info(f"  {i}. {path} ({match_type} match) - {status}")
                        if path and not is_duplicate:
                            new_paths.append(path)
                    logger.info(f"📝 Keyword search returning {len(new_paths)} new paths (filtered {len(keyword_results) - len(new_paths)} duplicates)")
                    return new_paths
                else:
                    logger.info("📝 Keyword search returned no results")
            else:
                logger.info("📝 Vault reader service not available")
        except Exception as e:
            logger.warning(f"📝 Keyword search failed: {e}")
        return []

    def _format_graph_output(self, query: str, graph: dict[str, Any], depth: int, search_type: str) -> str:
        """Format graph output for display."""
        response_lines = []

        # Header with summary
        response_lines.append(f"# Knowledge Graph for '{query}' ({search_type})")
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

        # Calculate unique relationship types
        unique_types = {
            rel.get('original_type', rel.get('type', 'UNKNOWN'))
            for rel in graph['relationships']
        }
        response_lines.append(f"- **Unique Relationship Types:** {len(unique_types)}")

        # Show relationship type distribution
        type_counts = {}
        for rel in graph['relationships']:
            rel_type = rel.get('original_type', rel.get('type', 'UNKNOWN'))
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        if type_counts:
            response_lines.append("- **Relationship Distribution:**")
            for rel_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                response_lines.append(f"  - {rel_type}: {count}")

        return "\n".join(response_lines)

    def _format_keyword_search_results(self, query: str, graphs: list[tuple], depth: int, all_note_paths: list[str]) -> str:
        """Format results when multiple notes are found via keyword search."""
        response_lines = []

        # Header
        response_lines.append(f"# Knowledge Graph Results for '{query}' (keyword search)")
        response_lines.append(f"**Found {len(all_note_paths)} matching notes, showing graphs for top {len(graphs)} results**")
        response_lines.append(f"**Traversal Depth:** {depth}")
        response_lines.append("")

        # Show all discovered notes
        response_lines.append("## 🔍 Discovered Notes")
        for i, path in enumerate(all_note_paths, 1):
            has_graph = any(graph_path == path for graph_path, _ in graphs)
            status = "📊 (graph shown below)" if has_graph else "📝 (no relationships found)"
            response_lines.append(f"{i}. **{path}** {status}")
        response_lines.append("")

        # Show graphs for each discovered note
        for i, (note_path, graph) in enumerate(graphs, 1):
            response_lines.append(f"## 📊 Graph {i}: {note_path}")
            response_lines.append(f"**Nodes:** {len(graph['nodes'])} | **Relationships:** {len(graph['relationships'])}")
            response_lines.append("")

            # Create node lookup
            node_lookup = {node['id']: node for node in graph['nodes']}

            # Group nodes by whether they are the center node
            center_nodes = [node for node in graph['nodes'] if node.get('center', False)]
            related_nodes = [node for node in graph['nodes'] if not node.get('center', False)]

            # Display center node
            if center_nodes:
                response_lines.append("### 🎯 Center Node")
                for node in center_nodes:
                    response_lines.append(f"**{node['label']}** (`{node['path']}`)")
                    if node.get('tags'):
                        response_lines.append(f"  Tags: {', '.join(node['tags'])}")

            # Display related nodes
            if related_nodes:
                response_lines.append("### 🔗 Connected Notes")
                for j, node in enumerate(related_nodes, 1):
                    response_lines.append(f"{j}. **{node['label']}** (`{node['path']}`)")
                    if node.get('tags'):
                        response_lines.append(f"   Tags: {', '.join(node['tags'])}")

            # Display relationships
            if graph['relationships']:
                response_lines.append("### 🌐 Relationships")

                # Group relationships by type
                relationships_by_type = {}
                for rel in graph['relationships']:
                    rel_type = rel.get('original_type', rel.get('type', 'UNKNOWN'))
                    if rel_type not in relationships_by_type:
                        relationships_by_type[rel_type] = []
                    relationships_by_type[rel_type].append(rel)

                for rel_type, rels in relationships_by_type.items():
                    response_lines.append(f"**{rel_type}:**")
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

        return "\n".join(response_lines)

    async def _fallback_to_semantic_search(self, query_note_path: str) -> list[types.TextContent]:
        """Fallback to semantic search when graph is unavailable."""
        try:
            if not self.container:
                return [types.TextContent(
                    type="text",
                    text="Graph search is unavailable and semantic search fallback cannot be accessed."
                )]

            searcher = self.container.get(IVectorSearcher)
            if not searcher:
                return [types.TextContent(
                    type="text",
                    text="Graph search is unavailable and semantic search fallback is not configured."
                )]

            # Perform semantic search using the note path as query
            results = searcher.search(query=query_note_path, top_k=FALLBACK_LIMIT)

            response_lines = [
                "**Graph search is unavailable. Falling back to semantic search.**\n",
                f"Found {len(results)} semantic results for '{query_note_path}':\n"
            ]

            for i, result in enumerate(results, 1):
                score = f"{result.similarity_score:.3f}"
                response_lines.append(
                    f"{i}. **{result.path}** (vault: {result.vault_name}, score: {score})"
                )

            return [types.TextContent(type="text", text="\n".join(response_lines))]

        except Exception as e:
            logger.error(f"Semantic search fallback failed: {e}")
            return [types.TextContent(
                type="text",
                text=f"Graph search is unavailable and semantic search fallback failed: {e!s}"
            )]
