"""
Caching system for vector search operations.

This module provides in-memory caching for search queries to improve
performance and reduce embedding computation overhead.
"""

import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.utils.logging import setup_logging

if TYPE_CHECKING:
    from jarvis.models.document import SearchResult

logger = setup_logging(__name__)


class QueryCache:
    """LRU cache for search queries and results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the query cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time to live for cached entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[list[SearchResult], float]] = OrderedDict()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_queries': 0
        }

        logger.info(f"Query cache initialized: max_size={max_size}, ttl={ttl_seconds}s")

    def _generate_key(
        self,
        query: str,
        top_k: int,
        vault_name: str | None = None,
        similarity_threshold: float | None = None
    ) -> str:
        """Generate a cache key for query parameters.
        
        Args:
            query: Search query
            top_k: Number of results
            vault_name: Optional vault filter
            similarity_threshold: Optional similarity threshold
            
        Returns:
            Cache key string
        """
        # Create a deterministic key from parameters
        key_data = f"{query}|{top_k}|{vault_name or ''}|{similarity_threshold or ''}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()[:16]

    def get(
        self,
        query: str,
        top_k: int,
        vault_name: str | None = None,
        similarity_threshold: float | None = None
    ) -> list["SearchResult"] | None:
        """Get cached search results.
        
        Args:
            query: Search query
            top_k: Number of results
            vault_name: Optional vault filter
            similarity_threshold: Optional similarity threshold
            
        Returns:
            Cached results or None if not found/expired
        """
        self._stats['total_queries'] += 1

        key = self._generate_key(query, top_k, vault_name, similarity_threshold)

        if key in self._cache:
            results, timestamp = self._cache[key]

            # Check if entry has expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._stats['misses'] += 1
                logger.debug(f"Cache entry expired: {query[:30]}")
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: {query[:30]}")
            return results

        self._stats['misses'] += 1
        logger.debug(f"Cache miss: {query[:30]}")
        return None

    def put(
        self,
        query: str,
        top_k: int,
        results: list["SearchResult"],
        vault_name: str | None = None,
        similarity_threshold: float | None = None
    ) -> None:
        """Store search results in cache.
        
        Args:
            query: Search query
            top_k: Number of results
            results: Search results to cache
            vault_name: Optional vault filter
            similarity_threshold: Optional similarity threshold
        """
        key = self._generate_key(query, top_k, vault_name, similarity_threshold)

        # Remove oldest entries if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats['evictions'] += 1

        # Store new entry
        self._cache[key] = (results, time.time())
        logger.debug(f"Cached results for: {query[:30]} ({len(results)} results)")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = 0.0
        if self._stats['total_queries'] > 0:
            hit_rate = self._stats['hits'] / self._stats['total_queries']

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'total_queries': self._stats['total_queries'],
            'hit_rate': hit_rate
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 10000):
        """Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

        logger.info(f"Embedding cache initialized: max_size={max_size}")

    def _generate_key(self, text: str) -> str:
        """Generate a cache key for text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key string
        """
        # Use first 1000 chars and hash for consistent key
        text_sample = text[:1000] if len(text) > 1000 else text
        return hashlib.md5(text_sample.encode('utf-8')).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Get cached embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_key(text)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            logger.debug(f"Embedding cache hit: {text[:30]}")
            return self._cache[key].copy()

        self._stats['misses'] += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        key = self._generate_key(text)

        # Remove oldest entries if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats['evictions'] += 1

        # Store new entry (copy to avoid modifications)
        self._cache[key] = embedding.copy()
        logger.debug(f"Cached embedding: {text[:30]}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = 0.0
        total_requests = self._stats['hits'] + self._stats['misses']
        if total_requests > 0:
            hit_rate = self._stats['hits'] / total_requests

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': hit_rate
        }
