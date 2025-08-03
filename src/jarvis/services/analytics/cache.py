"""
Multi-level analytics cache system.

This module implements a sophisticated caching system with three levels:
- L1: In-memory results (5 minutes TTL)
- L2: Computed analytics (1 hour TTL, invalidated on file changes)
- L3: Base statistics (24 hours TTL)
"""

import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from jarvis.services.analytics.models import CacheStatus, AnalyticsError
from jarvis.services.analytics.errors import CacheError
from jarvis.utils.config import get_settings


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    level: int  # 1, 2, or 3
    ttl_seconds: int
    vault_name: Optional[str] = None
    content_hash: Optional[str] = None  # For invalidation
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > (self.created_at + self.ttl_seconds)
    
    def is_stale(self, staleness_threshold: float = 0.8) -> bool:
        """Check if the cache entry is getting stale."""
        age = time.time() - self.created_at
        return age > (self.ttl_seconds * staleness_threshold)
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class AnalyticsCache:
    """
    Multi-level analytics cache with intelligent eviction and invalidation.
    
    Cache Levels:
    - L1 (5 min TTL): Recent results, fastest access
    - L2 (1 hour TTL): Computed analytics, invalidated on file changes
    - L3 (24 hour TTL): Base statistics, long-term cached
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analytics cache."""
        self.config = config or get_settings().get_analytics_config()["cache"]
        
        # Cache storage
        self._l1_cache: Dict[str, CacheEntry] = {}  # 5 min TTL
        self._l2_cache: Dict[str, CacheEntry] = {}  # 1 hour TTL  
        self._l3_cache: Dict[str, CacheEntry] = {}  # 24 hours TTL
        
        # Cache configuration
        self._max_size_mb: int = self.config.get("max_size_mb", 100)
        self._base_ttl_minutes: int = self.config.get("ttl_minutes", 60)
        
        # TTL configuration (in seconds)
        self._l1_ttl = 5 * 60      # 5 minutes
        self._l2_ttl = 60 * 60     # 1 hour
        self._l3_ttl = 24 * 60 * 60  # 24 hours
        
        # Performance tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "cleanup_runs": 0,
            "errors": 0,
            "total_lookups": 0,
            "lookup_times": [],
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-cleanup")
        
        # Vault content tracking for invalidation
        self._vault_content_hashes: Dict[str, str] = {}
        self._cache_key_mappings: Dict[str, Set[str]] = defaultdict(set)  # vault -> cache keys
        
        # Start background cleanup
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info(f"Analytics cache initialized with {self._max_size_mb}MB limit")
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            logger.warning("No event loop for background cache cleanup")
    
    def _generate_cache_key(self, operation: str, vault_name: str, **kwargs) -> str:
        """Generate a deterministic cache key."""
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items()) if kwargs else []
        key_data = f"{operation}:{vault_name}:{sorted_kwargs}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _calculate_size_bytes(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            if isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode())
            elif isinstance(value, str):
                return len(value.encode())
            elif hasattr(value, '__dict__'):
                return len(json.dumps(asdict(value), default=str).encode())
            else:
                return len(str(value).encode())
        except Exception:
            return 1024  # Default estimate
    
    def _get_cache_level(self, cache_level: int) -> Dict[str, CacheEntry]:
        """Get the appropriate cache level dictionary."""
        if cache_level == 1:
            return self._l1_cache
        elif cache_level == 2:
            return self._l2_cache
        elif cache_level == 3:
            return self._l3_cache
        else:
            raise ValueError(f"Invalid cache level: {cache_level}")
    
    def _get_ttl_for_level(self, cache_level: int) -> int:
        """Get TTL in seconds for a cache level."""
        if cache_level == 1:
            return self._l1_ttl
        elif cache_level == 2:
            return self._l2_ttl
        elif cache_level == 3:
            return self._l3_ttl
        else:
            raise ValueError(f"Invalid cache level: {cache_level}")
    
    def get(self, operation: str, vault_name: str, cache_level: int = 2, **kwargs) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            operation: The operation name (e.g., "vault_context", "quality_analysis")
            vault_name: The vault name
            cache_level: Which cache level to check (1, 2, or 3)
            **kwargs: Additional parameters for cache key generation
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                cache_key = self._generate_cache_key(operation, vault_name, **kwargs)
                
                # Check the specified level first, then fall back to other levels
                for level in [cache_level, 1, 2, 3]:
                    if level == cache_level:
                        continue  # Already checked
                    
                    cache_dict = self._get_cache_level(level)
                    entry = cache_dict.get(cache_key)
                    
                    if entry and not entry.is_expired():
                        entry.touch()
                        self._stats["hits"] += 1
                        self._stats["total_lookups"] += 1
                        lookup_time = (time.time() - start_time) * 1000
                        self._stats["lookup_times"].append(lookup_time)
                        
                        logger.debug(f"Cache hit L{level}: {cache_key}")
                        return entry.value
                
                # Cache miss
                self._stats["misses"] += 1
                self._stats["total_lookups"] += 1
                lookup_time = (time.time() - start_time) * 1000
                self._stats["lookup_times"].append(lookup_time)
                
                logger.debug(f"Cache miss: {cache_key}")
                return None
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            raise CacheError("get", cache_key, str(e))
    
    def set(
        self,
        operation: str,
        vault_name: str,
        value: Any,
        cache_level: int = 2,
        content_hash: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            operation: The operation name
            vault_name: The vault name  
            value: The value to cache
            cache_level: Which cache level to use (1, 2, or 3)
            content_hash: Content hash for invalidation
            **kwargs: Additional parameters for cache key generation
        
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            with self._lock:
                cache_key = self._generate_cache_key(operation, vault_name, **kwargs)
                cache_dict = self._get_cache_level(cache_level)
                ttl = self._get_ttl_for_level(cache_level)
                
                # Calculate size
                size_bytes = self._calculate_size_bytes(value)
                
                # Check if adding this entry would exceed size limit
                if self._would_exceed_size_limit(size_bytes):
                    self._evict_lru()
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    level=cache_level,
                    ttl_seconds=ttl,
                    vault_name=vault_name,
                    content_hash=content_hash
                )
                
                # Store in cache
                cache_dict[cache_key] = entry
                
                # Track for invalidation
                self._cache_key_mappings[vault_name].add(cache_key)
                if content_hash:
                    self._vault_content_hashes[vault_name] = content_hash
                
                logger.debug(f"Cache set L{cache_level}: {cache_key} ({size_bytes} bytes)")
                return True
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            raise CacheError("set", cache_key, str(e))
    
    def invalidate(self, vault_name: Optional[str] = None, operation: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            vault_name: If specified, invalidate only entries for this vault
            operation: If specified, invalidate only entries for this operation
        
        Returns:
            Number of entries invalidated
        """
        try:
            with self._lock:
                invalidated_count = 0
                
                if vault_name:
                    # Invalidate all entries for a specific vault
                    cache_keys = self._cache_key_mappings.get(vault_name, set()).copy()
                    
                    for cache_key in cache_keys:
                        # Remove from all cache levels
                        for level in [1, 2, 3]:
                            cache_dict = self._get_cache_level(level)
                            if cache_key in cache_dict:
                                del cache_dict[cache_key]
                                invalidated_count += 1
                    
                    # Clean up mappings
                    self._cache_key_mappings[vault_name].clear()
                    if vault_name in self._vault_content_hashes:
                        del self._vault_content_hashes[vault_name]
                
                else:
                    # Invalidate all entries
                    for level in [1, 2, 3]:
                        cache_dict = self._get_cache_level(level)
                        invalidated_count += len(cache_dict)
                        cache_dict.clear()
                    
                    self._cache_key_mappings.clear()
                    self._vault_content_hashes.clear()
                
                self._stats["invalidations"] += invalidated_count
                
                logger.info(f"Invalidated {invalidated_count} cache entries" + 
                           (f" for vault '{vault_name}'" if vault_name else ""))
                
                return invalidated_count
                
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cache invalidation error: {e}")
            raise CacheError("invalidate", vault_name or "all", str(e))
    
    def _would_exceed_size_limit(self, additional_bytes: int) -> bool:
        """Check if adding additional bytes would exceed the size limit."""
        current_size = self.get_current_size_mb()
        new_size_mb = (current_size * 1024 * 1024 + additional_bytes) / (1024 * 1024)
        return new_size_mb > self._max_size_mb
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries to make space."""
        # Collect all entries across all levels
        all_entries: List[Tuple[CacheEntry, int]] = []
        
        for level in [1, 2, 3]:
            cache_dict = self._get_cache_level(level)
            for entry in cache_dict.values():
                all_entries.append((entry, level))
        
        # Sort by last accessed time (oldest first)
        all_entries.sort(key=lambda x: x[0].last_accessed)
        
        # Evict oldest entries until we're under the limit
        evicted = 0
        current_size_bytes = self.get_current_size_mb() * 1024 * 1024
        target_size_bytes = self._max_size_mb * 1024 * 1024 * 0.8  # Target 80% of limit
        
        for entry, level in all_entries:
            if current_size_bytes <= target_size_bytes:
                break
            
            cache_dict = self._get_cache_level(level)
            if entry.key in cache_dict:
                del cache_dict[entry.key]
                current_size_bytes -= entry.size_bytes
                evicted += 1
                
                # Clean up mappings
                if entry.vault_name:
                    self._cache_key_mappings[entry.vault_name].discard(entry.key)
        
        self._stats["evictions"] += evicted
        logger.info(f"Evicted {evicted} cache entries to manage size")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from all cache levels."""
        try:
            with self._lock:
                total_removed = 0
                
                for level in [1, 2, 3]:
                    cache_dict = self._get_cache_level(level)
                    expired_keys = [
                        key for key, entry in cache_dict.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        entry = cache_dict[key]
                        del cache_dict[key]
                        total_removed += 1
                        
                        # Clean up mappings
                        if entry.vault_name:
                            self._cache_key_mappings[entry.vault_name].discard(key)
                
                self._stats["cleanup_runs"] += 1
                
                if total_removed > 0:
                    logger.debug(f"Cleaned up {total_removed} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def get_current_size_mb(self) -> float:
        """Get current cache size in MB."""
        total_bytes = 0
        
        for level in [1, 2, 3]:
            cache_dict = self._get_cache_level(level)
            total_bytes += sum(entry.size_bytes for entry in cache_dict.values())
        
        return total_bytes / (1024 * 1024)
    
    def get_cache_status(self) -> CacheStatus:
        """Get comprehensive cache status information."""
        with self._lock:
            current_time = time.time()
            
            # Calculate hit rate
            total_requests = self._stats["hits"] + self._stats["misses"]
            cache_hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            # Count entries by level
            l1_entries = len(self._l1_cache)
            l2_entries = len(self._l2_cache)
            l3_entries = len(self._l3_cache)
            total_entries = l1_entries + l2_entries + l3_entries
            
            # Find oldest entry
            oldest_age_minutes = 0.0
            if total_entries > 0:
                all_entries = (
                    list(self._l1_cache.values()) +
                    list(self._l2_cache.values()) +
                    list(self._l3_cache.values())
                )
                oldest_entry = min(all_entries, key=lambda e: e.created_at)
                oldest_age_minutes = (current_time - oldest_entry.created_at) / 60
            
            # Calculate average lookup time
            lookup_times = self._stats["lookup_times"]
            avg_lookup_time = sum(lookup_times) / len(lookup_times) if lookup_times else 0.0
            
            # Calculate cache efficiency (hit rate weighted by recency)
            cache_efficiency = min(cache_hit_rate * 1.2, 1.0) if cache_hit_rate > 0 else 0.0
            
            return CacheStatus(
                cache_hit_rate=cache_hit_rate,
                total_entries=total_entries,
                memory_usage_mb=self.get_current_size_mb(),
                oldest_entry_age_minutes=oldest_age_minutes,
                l1_entries=l1_entries,
                l2_entries=l2_entries,
                l3_entries=l3_entries,
                average_lookup_time_ms=avg_lookup_time,
                cache_efficiency_score=cache_efficiency,
                last_cleanup=current_time,
            )
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics for monitoring."""
        with self._lock:
            return {
                **self._stats,
                "cache_levels": {
                    "l1": {
                        "entries": len(self._l1_cache),
                        "ttl_seconds": self._l1_ttl,
                        "size_mb": sum(e.size_bytes for e in self._l1_cache.values()) / (1024 * 1024)
                    },
                    "l2": {
                        "entries": len(self._l2_cache),
                        "ttl_seconds": self._l2_ttl,
                        "size_mb": sum(e.size_bytes for e in self._l2_cache.values()) / (1024 * 1024)
                    },
                    "l3": {
                        "entries": len(self._l3_cache),
                        "ttl_seconds": self._l3_ttl,
                        "size_mb": sum(e.size_bytes for e in self._l3_cache.values()) / (1024 * 1024)
                    }
                },
                "configuration": {
                    "max_size_mb": self._max_size_mb,
                    "base_ttl_minutes": self._base_ttl_minutes,
                },
                "vault_mappings": {
                    vault: len(keys) for vault, keys in self._cache_key_mappings.items()
                }
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
            self._l3_cache.clear()
            self._cache_key_mappings.clear()
            self._vault_content_hashes.clear()
            
            # Reset stats
            self._stats = {key: 0 if isinstance(val, (int, float)) else [] 
                          for key, val in self._stats.items()}
            
            logger.info("Analytics cache cleared")
    
    def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if self._cleanup_executor:
            self._cleanup_executor.shutdown(wait=True)
        
        self.clear()
        logger.info("Analytics cache closed")