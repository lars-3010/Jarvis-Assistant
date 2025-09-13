"""
Vector search service for semantic document retrieval.

This module provides high-level search functionality that combines
the vector database and encoder to perform semantic search operations.
"""

from pathlib import Path
from typing import Any

import torch

from jarvis.core.interfaces import IVectorSearcher
from jarvis.models.document import SearchResult
from jarvis.services.vector.cache import QueryCache
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.config import get_settings
from jarvis.utils.errors import ServiceError
import logging

logger = logging.getLogger(__name__)


class VectorSearcher(IVectorSearcher):
    """High-level semantic search interface with caching."""

    def __init__(
        self,
        database: VectorDatabase,
        encoder: VectorEncoder,
        vaults: dict[str, Path],
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """Initialize the searcher.
        
        Args:
            database: Vector database instance
            encoder: Vector encoder instance  
            vaults: Dictionary mapping vault names to paths
            enable_cache: Whether to enable query caching
            cache_size: Maximum number of cached queries
            cache_ttl: Cache time-to-live in seconds
        """
        self.database = database
        self.encoder = encoder
        self.vaults = vaults

        # Initialize cache if enabled
        self.cache = QueryCache(cache_size, cache_ttl) if enable_cache else None

        # Performance metrics
        self._search_stats = {
            'total_searches': 0,
            'total_search_time': 0.0,
            'cache_enabled': enable_cache
        }

        logger.info(f"Initialized searcher with {len(vaults)} vaults, cache={'enabled' if enable_cache else 'disabled'}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        vault_name: str | None = None,
        similarity_threshold: float | None = None
    ) -> list[SearchResult]:
        """Search for notes that match a query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            vault_name: Optional vault name filter
            similarity_threshold: Minimum similarity score (from settings if None)
            
        Returns:
            List of SearchResult objects
        """
        import time
        start_time = time.time()

        if not query or not query.strip():
            logger.warning("Empty query provided to search")
            return []

        # Get similarity threshold from settings if not provided
        if similarity_threshold is None:
            settings = get_settings()
            similarity_threshold = getattr(settings, 'search_similarity_threshold', 0.0)

        # Check cache first
        if self.cache:
            cached_results = self.cache.get(query, top_k, vault_name, similarity_threshold)
            if cached_results is not None:
                self._update_search_stats(time.time() - start_time, True)
                return cached_results

        try:
            # Encode the query
            query_embedding = self.encoder.encode_query(query)

            if query_embedding.size == 0:
                logger.error("Failed to encode query, received empty embedding")
                raise ServiceError("Failed to encode query, received empty embedding")

            # Convert to torch tensor for database search
            query_tensor = torch.from_numpy(query_embedding)

            # Search the database
            raw_results = self.database.search(
                query_tensor,
                top_k=top_k,
                vault_name=vault_name
            )

            # Convert to SearchResult objects and apply threshold
            search_results = []
            for vault, path, score in raw_results:
                if score >= similarity_threshold:
                    # Resolve full path if vault is known
                    full_path = self.vaults.get(vault, Path()) / path
                    result = SearchResult(vault, path, score, full_path)
                    search_results.append(result)

            # Cache the results
            if self.cache:
                self.cache.put(query, top_k, search_results, vault_name, similarity_threshold)

            self._update_search_stats(time.time() - start_time, False)
            logger.debug(f"Search for '{query[:50]}...': returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise ServiceError(f"Search failed for query '{query[:50]}...': {e}") from e

    def search_similar(
        self,
        document_text: str,
        top_k: int = 10,
        vault_name: str | None = None,
        exclude_self: bool = True
    ) -> list[SearchResult]:
        """Find documents similar to the given document text.
        
        Args:
            document_text: Text of the document to find similar documents for
            top_k: Maximum number of results to return
            vault_name: Optional vault name filter
            exclude_self: Whether to exclude the document itself from results
            
        Returns:
            List of SearchResult objects
        """
        # Use the same search logic but with document text as query
        return self.search(
            query=document_text[:1000],  # Use first 1000 chars as query
            top_k=top_k + (1 if exclude_self else 0),  # Get extra if excluding self
            vault_name=vault_name
        )

    def get_paths_only(
        self,
        query: str,
        top_k: int = 10,
        vault_name: str | None = None
    ) -> list[Path]:
        """Search and return only the full file paths (backwards compatibility).
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            vault_name: Optional vault name filter
            
        Returns:
            List of full file paths
        """
        results = self.search(query, top_k, vault_name)
        return [result.full_path for result in results]

    def search_by_vault(
        self,
        query: str,
        top_k: int = 10
    ) -> dict[str, list[SearchResult]]:
        """Search across all vaults and group results by vault.
        
        Args:
            query: Search query text
            top_k: Maximum number of results per vault
            
        Returns:
            Dictionary mapping vault names to search results
        """
        results_by_vault = {}

        for vault_name in self.vaults.keys():
            vault_results = self.search(
                query=query,
                top_k=top_k,
                vault_name=vault_name
            )
            if vault_results:
                results_by_vault[vault_name] = vault_results

        return results_by_vault

    def get_vault_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all configured vaults.
        
        Returns:
            Dictionary mapping vault names to their statistics
        """
        stats = {}
        for vault_name in self.vaults.keys():
            stats[vault_name] = self.database.get_vault_stats(vault_name)
        return stats

    def validate_vaults(self) -> dict[str, bool]:
        """Validate that all configured vault paths exist.
        
        Returns:
            Dictionary mapping vault names to existence status
        """
        validation = {}
        for vault_name, vault_path in self.vaults.items():
            validation[vault_name] = vault_path.exists() and vault_path.is_dir()
            if not validation[vault_name]:
                logger.warning(f"Vault path does not exist: {vault_name} -> {vault_path}")
        return validation

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the search configuration.
        
        Returns:
            Dictionary with search and model information
        """
        info = {
            'encoder_info': self.encoder.get_model_info(),
            'vault_count': len(self.vaults),
            'vault_names': list(self.vaults.keys()),
            'database_note_count': self.database.num_notes(),
            'cache_enabled': self.cache is not None,
            'search_stats': self._search_stats.copy()
        }

        if self.cache:
            info['cache_stats'] = self.cache.get_stats()

        return info

    def _update_search_stats(self, search_time: float, was_cached: bool) -> None:
        """Update search performance statistics.
        
        Args:
            search_time: Time taken for the search
            was_cached: Whether the result came from cache
        """
        self._search_stats['total_searches'] += 1
        if not was_cached:  # Only count actual search time, not cache lookups
            self._search_stats['total_search_time'] += search_time

    def get_search_stats(self) -> dict[str, Any]:
        """Get detailed search performance statistics.
        
        Returns:
            Dictionary with search performance statistics
        """
        stats = self._search_stats.copy()

        if stats['total_searches'] > 0:
            cache_hits = 0
            if self.cache:
                cache_stats = self.cache.get_stats()
                cache_hits = cache_stats['hits']

            stats['cache_hit_rate'] = cache_hits / stats['total_searches']

            # Calculate average search time (excluding cached searches)
            actual_searches = stats['total_searches'] - cache_hits
            if actual_searches > 0:
                stats['avg_search_time'] = stats['total_search_time'] / actual_searches
            else:
                stats['avg_search_time'] = 0.0

        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the search cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Search cache cleared")

    def cleanup_cache(self) -> int:
        """Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        if self.cache:
            return self.cache.cleanup_expired()
        return 0
