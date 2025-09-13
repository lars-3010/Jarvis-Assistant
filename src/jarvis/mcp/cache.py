"""
Caching system for MCP tool operations.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


class MCPToolCache:
    """LRU cache for MCP tool calls and their results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize the MCP tool cache.

        Args:
            max_size: Maximum number of cached tool call results.
            ttl_seconds: Time to live for cached entries in seconds.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[list[types.TextContent | types.ImageContent | types.EmbeddedResource], float]] = OrderedDict()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        logger.info(f"MCP tool cache initialized: max_size={max_size}, ttl={ttl_seconds}s")

    def _generate_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Generate a cache key for a tool call.
        """
        key_data = {"tool": tool_name, "args": arguments}
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode('utf-8')).hexdigest()

    def get(self, tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource] | None:
        """
        Retrieve cached results for a tool call.
        """
        with self._lock:
            self._stats['total_requests'] += 1
            key = self._generate_key(tool_name, arguments)

            if key in self._cache:
                results, timestamp = self._cache[key]

                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    self._stats['misses'] += 1
                    logger.debug(f"MCP cache entry expired for tool {tool_name}")
                    return None

                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                logger.debug(f"MCP cache hit for tool {tool_name}")
                return results

            self._stats['misses'] += 1
            logger.debug(f"MCP cache miss for tool {tool_name}")
            return None

    def put(self, tool_name: str, arguments: dict[str, Any], results: list[types.TextContent | types.ImageContent | types.EmbeddedResource]) -> None:
        """
        Store results of a tool call in the cache.
        """
        with self._lock:
            key = self._generate_key(tool_name, arguments)

            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats['evictions'] += 1

            self._cache[key] = (results, time.time())
            logger.debug(f"Cached results for tool {tool_name}")

    def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        with self._lock:
            self._cache.clear()
            self._stats['hits'] = 0
            self._stats['misses'] = 0
            self._stats['evictions'] = 0
            self._stats['total_requests'] = 0
            logger.info("MCP tool cache cleared.")

    def get_stats(self) -> dict[str, Any]:
        """
        Get current cache statistics.
        """
        with self._lock:
            hit_rate = 0.0
            if self._stats['total_requests'] > 0:
                hit_rate = self._stats['hits'] / self._stats['total_requests']

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'total_requests': self._stats['total_requests'],
                'hit_rate': hit_rate
            }
