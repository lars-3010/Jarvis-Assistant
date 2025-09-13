"""
Search Combined Plugin for MCP Tools.

This plugin provides combined semantic and keyword search capabilities
across vault content with intelligent result fusion and unified scoring.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from jarvis.core.interfaces import IVaultReader, IVectorSearcher
from jarvis.mcp.plugins.base import SearchPlugin
from jarvis.mcp.structured import combined_search_to_json
from jarvis.mcp.schemas import SearchSchemaConfig, create_search_schema
from jarvis.utils.errors import PluginError
import logging
from mcp import types

logger = logging.getLogger(__name__)

# Constants for result fusion
HIGH_SCORE_THRESHOLD = 0.8
SEMANTIC_HIGH_THRESHOLD = -2
SEMANTIC_MEDIUM_THRESHOLD = -5
QUERY_PREVIEW_LENGTH = 50


@dataclass
class UnifiedResult:
    """Unified result with normalized scoring and context."""
    path: str
    vault_name: str
    unified_score: float
    semantic_score: float | None = None
    keyword_score: float | None = None
    match_type: str | None = None
    context_snippet: str | None = None
    match_reasons: list[str] = None

    def __post_init__(self):
        if self.match_reasons is None:
            self.match_reasons = []


class ResultFusion:
    """Handles intelligent result fusion with unified scoring."""

    def __init__(self):
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        self.exact_match_boost = 0.2
        self.filename_match_boost = 0.1

    def normalize_semantic_score(self, score: float) -> float:
        """Normalize semantic similarity score to 0-1 range."""
        # Semantic scores are typically negative (closer to 0 = more similar)
        # Convert to 0-1 range where 1 = most similar
        if score >= 0:
            return max(0, 1 - score)
        else:
            # For negative scores, use exponential decay
            return max(0, math.exp(score / 5))

    def score_keyword_match(self, match_type: str, query: str, path: str) -> float:
        """Score keyword matches based on type and context."""
        base_scores = {
            'exact': 0.9,
            'filename': 0.8,
            'content': 0.6,
            'name': 0.7,
            'fuzzy': 0.5,
            'children': 0.4
        }

        base_score = base_scores.get(match_type, 0.5)

        # Boost for exact filename matches
        if match_type == 'filename' and query.lower() in Path(path).stem.lower():
            base_score += self.filename_match_boost

        # Boost for exact path matches
        if query.lower() in path.lower():
            base_score += self.exact_match_boost

        return min(1.0, base_score)

    def calculate_unified_score(self, semantic_score: float | None,
                              keyword_score: float | None) -> float:
        """Calculate unified score from semantic and keyword scores."""
        if semantic_score is not None and keyword_score is not None:
            # Both scores available - weighted average
            normalized_semantic = self.normalize_semantic_score(semantic_score)
            return (normalized_semantic * self.semantic_weight +
                   keyword_score * self.keyword_weight)
        elif semantic_score is not None:
            # Only semantic score available
            return self.normalize_semantic_score(semantic_score) * 0.9
        elif keyword_score is not None:
            # Only keyword score available
            return keyword_score * 0.8
        else:
            # No scores available
            return 0.0

    def deduplicate_results(self, results: list[UnifiedResult]) -> list[UnifiedResult]:
        """Remove duplicates and merge match reasons."""
        seen_paths = {}
        deduplicated = []

        for result in results:
            if result.path in seen_paths:
                # Merge with existing result
                existing = seen_paths[result.path]

                # Use higher unified score
                if result.unified_score > existing.unified_score:
                    existing.unified_score = result.unified_score
                    existing.semantic_score = result.semantic_score or existing.semantic_score
                    existing.keyword_score = result.keyword_score or existing.keyword_score

                # Merge match reasons
                if result.match_reasons:
                    existing.match_reasons.extend(result.match_reasons)
                    existing.match_reasons = list(set(existing.match_reasons))

                # Update match type if more specific
                if result.match_type == 'exact' or (result.match_type == 'filename' and existing.match_type == 'content'):
                    existing.match_type = result.match_type

            else:
                seen_paths[result.path] = result
                deduplicated.append(result)

        return deduplicated

    def diversify_results(self, results: list[UnifiedResult], max_per_folder: int = 3) -> list[UnifiedResult]:
        """Diversify results to avoid clustering in single folders."""
        folder_counts = {}
        diversified = []

        for result in results:
            folder = str(Path(result.path).parent)
            folder_count = folder_counts.get(folder, 0)

            if folder_count < max_per_folder:
                diversified.append(result)
                folder_counts[folder] = folder_count + 1
            elif result.unified_score > HIGH_SCORE_THRESHOLD:  # Keep high-scoring results regardless
                diversified.append(result)

        return diversified

    def generate_context_snippet(self, result: UnifiedResult, query: str) -> str:
        """Generate context snippet explaining why result matched."""
        reasons = []

        if result.semantic_score is not None:
            if result.semantic_score > SEMANTIC_HIGH_THRESHOLD:
                semantic_quality = "high"
            elif result.semantic_score > SEMANTIC_MEDIUM_THRESHOLD:
                semantic_quality = "medium"
            else:
                semantic_quality = "low"
            reasons.append(f"semantic similarity: {semantic_quality}")

        if result.keyword_score is not None and result.match_type:
            reasons.append(f"{result.match_type} match")

        if result.match_reasons:
            reasons.extend(result.match_reasons)

        return f"Found via: {', '.join(reasons)}"


class SearchCombinedPlugin(SearchPlugin):
    """Plugin for combined semantic and keyword search."""

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return "search-combined"

    @property
    def description(self) -> str:
        """Get the plugin description."""
        return "Perform a combined semantic and keyword search across vault content"

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
        return ["search", "semantic", "keyword", "combined"]

    def get_required_services(self) -> list[type]:
        """Get required service interfaces."""
        return [IVectorSearcher, IVaultReader]

    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition using standardized schema."""
        schema_config = SearchSchemaConfig(
            enable_content_search=True,
            enable_vault_selection=True,
            default_limit=10,
            max_limit=50,
            supported_formats=["json"],
        )

        input_schema = create_search_schema(schema_config)

        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=input_schema,
        )

    def _parse_arguments(self, arguments: dict[str, Any]) -> tuple[str, int, bool, str | None, str]:
        """Parse and validate arguments."""
        query = arguments.get("query", "").strip()
        limit = arguments.get("limit", 10)
        search_content = arguments.get("search_content", True)
        vault_name = arguments.get("vault")
        output_format = arguments.get("format", "markdown").lower()

        query_preview = f"{query[:QUERY_PREVIEW_LENGTH]}{'...' if len(query) > QUERY_PREVIEW_LENGTH else ''}"
        logger.info(f"📋 Parsed arguments - query: '{query_preview}', limit: {limit}, search_content: {search_content}, vault_name: {vault_name}")

        return query, limit, search_content, vault_name, output_format

    def _get_services(self) -> tuple[IVectorSearcher, IVaultReader]:
        """Get required services from container."""
        if not self.container:
            raise PluginError("Service container not available")

        searcher = self.container.get(IVectorSearcher)
        vault_reader = self.container.get(IVaultReader)

        if not searcher:
            raise PluginError("Vector searcher service not available")
        if not vault_reader:
            raise PluginError("Vault reader service not available")

        return searcher, vault_reader

    def _perform_searches(self, query: str, limit: int, search_content: bool, vault_name: str | None,
                         searcher: IVectorSearcher, vault_reader: IVaultReader) -> tuple[list, list]:
        """Perform both semantic and keyword searches."""
        # Semantic search
        semantic_results = []
        try:
            semantic_results = searcher.search(
                query=query,
                top_k=limit * 2,
                vault_name=vault_name
            )
            logger.info(f"✅ Semantic search found {len(semantic_results)} results")
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")

        # Keyword search
        keyword_results = []
        try:
            keyword_results = vault_reader.search_vault(
                query=query,
                search_content=search_content,
                limit=limit * 2
            )
            logger.info(f"✅ Keyword search found {len(keyword_results)} results")
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")

        return semantic_results, keyword_results

    def _convert_to_unified_results(self, semantic_results: list, keyword_results: list,
                                   query: str, vault_name: str | None, fusion: ResultFusion) -> list[UnifiedResult]:
        """Convert search results to unified format."""
        unified_results = []

        # Process semantic results
        for result in semantic_results or []:
            vault_name_result = getattr(result, 'vault_name', vault_name or 'default')
            unified_result = UnifiedResult(
                path=result.path,
                vault_name=vault_name_result,
                semantic_score=result.similarity_score,
                unified_score=0.0,
                match_reasons=["semantic search"]
            )
            unified_results.append(unified_result)

        # Process keyword results
        for result in keyword_results or []:
            if isinstance(result, dict):
                path = result.get('path', 'Unknown')
                match_type = result.get('match_type', 'name')
            else:
                path = getattr(result, 'path', 'Unknown')
                match_type = getattr(result, 'match_type', 'name')

            keyword_score = fusion.score_keyword_match(match_type, query, path)
            unified_result = UnifiedResult(
                path=path,
                vault_name=vault_name or 'default',
                keyword_score=keyword_score,
                match_type=match_type,
                unified_score=0.0,
                match_reasons=["keyword search"]
            )
            unified_results.append(unified_result)

        return unified_results

    def _format_results(self, unified_results: list[UnifiedResult], query: str, fusion: ResultFusion) -> list[str]:
        """Format results for display."""
        response_lines = [f"# Enhanced Combined Search Results for '{query}'\n"]
        response_lines.append(f"**Found {len(unified_results)} results (ranked by unified relevance)**\n")

        if not unified_results:
            response_lines.append("No results found.")
        else:
            for i, result in enumerate(unified_results, 1):
                # Format score display
                score_parts = []
                if result.semantic_score is not None:
                    normalized_sem = fusion.normalize_semantic_score(result.semantic_score)
                    score_parts.append(f"semantic: {normalized_sem:.3f}")
                if result.keyword_score is not None:
                    score_parts.append(f"keyword: {result.keyword_score:.3f}")

                score_display = " | ".join(score_parts) if score_parts else "no scores"

                response_lines.append(f"{i}. **{result.path}** (vault: {result.vault_name})")
                response_lines.append(f"   *Relevance: {result.unified_score:.3f} ({score_display})*")
                response_lines.append(f"   *{result.context_snippet}*")
                response_lines.append("")

        return response_lines

    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute combined search with intelligent result fusion."""
        start_time = time.time()

        logger.info("🔄 Starting enhanced combined search execution")
        logger.debug(f"📝 Raw arguments received: {arguments}")

        # Parse and validate arguments
        query, limit, search_content, vault_name, output_format = self._parse_arguments(arguments)

        if not query:
            logger.warning("❌ Query validation failed: empty query provided")
            return [types.TextContent(
                type="text",
                text="Error: Query parameter is required and cannot be empty"
            )]

        try:
            # Get services
            searcher, vault_reader = self._get_services()
            logger.info("✅ All required services are available")

            # Initialize result fusion
            fusion = ResultFusion()

            # Perform searches
            semantic_results, keyword_results = self._perform_searches(
                query, limit, search_content, vault_name, searcher, vault_reader
            )

            # Convert to unified format
            unified_results = self._convert_to_unified_results(
                semantic_results, keyword_results, query, vault_name, fusion
            )

            # Calculate unified scores
            logger.info("📊 Calculating unified scores")
            for result in unified_results:
                result.unified_score = fusion.calculate_unified_score(
                    result.semantic_score, result.keyword_score
                )

            # Apply result fusion pipeline
            logger.info("🔧 Applying result fusion pipeline")
            unified_results = fusion.deduplicate_results(unified_results)
            unified_results.sort(key=lambda x: x.unified_score, reverse=True)
            unified_results = fusion.diversify_results(unified_results)
            unified_results = unified_results[:limit]

            # Generate context snippets
            for result in unified_results:
                result.context_snippet = fusion.generate_context_snippet(result, query)

            total_duration = time.time() - start_time
            logger.info(f"🎉 Enhanced combined search completed successfully in {total_duration:.3f}s")

            if output_format == "json":
                payload = combined_search_to_json(
                    query=query,
                    unified_results=unified_results,
                    execution_time_ms=int(total_duration * 1000),
                )
                payload["correlation_id"] = str(uuid4())
                return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]
            else:
                # Fallback to JSON if an unsupported format is requested
                payload = combined_search_to_json(
                    query=query,
                    unified_results=unified_results,
                    execution_time_ms=int(total_duration * 1000),
                )
                payload["correlation_id"] = str(uuid4())
                return [types.TextContent(type="text", text=json.dumps(payload, indent=2))]

        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"💥 Enhanced combined search error after {total_duration:.3f}s: {e}")
            raise PluginError(f"Enhanced combined search failed: {e!s}") from e
