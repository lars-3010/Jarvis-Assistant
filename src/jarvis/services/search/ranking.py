"""
Service for ranking and deduplicating search results.
"""

from collections import OrderedDict
from typing import Any

from jarvis.models.document import SearchResult
import logging

logger = logging.getLogger(__name__)


class ResultRanker:
    """Handles ranking and deduplication of various search result types."""

    def rank_results(self, results: list[SearchResult | dict[str, Any]]) -> list[SearchResult | dict[str, Any]]:
        """
        Ranks a list of search results. Semantic results are prioritized by score,
        keyword results are sorted alphabetically by path if no score is available.
        """
        def get_sort_key(item):
            if isinstance(item, SearchResult):
                return (0, -item.similarity_score)  # Semantic results, higher score first (priority 0)
            elif isinstance(item, dict) and "path" in item:
                # For keyword results, lower priority (priority 1) and sort by path
                return (1, item["path"])
            return (2, "") # Fallback for unexpected types, lowest priority

        return sorted(results, key=get_sort_key)

    def deduplicate_results(self, results: list[SearchResult | dict[str, Any]]) -> list[SearchResult | dict[str, Any]]:
        """
        Removes duplicate results based on their unique identifier (path + vault_name).
        Prioritizes semantic results over keyword results if paths are identical.
        """
        unique_results = OrderedDict() # Use OrderedDict to maintain insertion order

        for item in results:
            if isinstance(item, SearchResult):
                unique_id = f"{item.vault_name}::{item.path}"
                if unique_id not in unique_results or unique_results[unique_id]["type"] == "keyword":
                    unique_results[unique_id] = {"type": "semantic", "data": item}
            elif isinstance(item, dict) and "path" in item:
                vault_name = item.get("vault_name", "default") # Assume default vault if not specified
                unique_id = f"{vault_name}::{item['path']}"
                if unique_id not in unique_results:
                    unique_results[unique_id] = {"type": "keyword", "data": item}

        # Convert back to list of original objects/dicts
        return [item["data"] for item in unique_results.values()]

    def merge_and_rank(
        self,
        semantic_results: list[SearchResult],
        keyword_results: list[dict[str, Any]]
    ) -> list[SearchResult | dict[str, Any]]:
        """
        Merges semantic and keyword search results, deduplicates, and ranks them.
        """
        # Combine all results
        all_results = semantic_results + keyword_results

        # Deduplicate, prioritizing semantic results
        deduplicated = self.deduplicate_results(all_results)

        # Rank the deduplicated results
        ranked = self.rank_results(deduplicated)

        logger.debug(f"Merged {len(semantic_results)} semantic and {len(keyword_results)} keyword results into {len(ranked)} ranked results.")
        return ranked
