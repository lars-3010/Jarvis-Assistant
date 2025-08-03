import pytest
import time
from unittest.mock import MagicMock

from jarvis.mcp.cache import MCPToolCache
import mcp.types as types

@pytest.fixture
def mcp_cache():
    return MCPToolCache(max_size=2, ttl_seconds=1)

def test_cache_put_and_get(mcp_cache):
    tool_name = "test-tool"
    arguments = {"param1": "value1"}
    results = [types.TextContent(type="text", text="Test result")]

    mcp_cache.put(tool_name, arguments, results)
    cached_results = mcp_cache.get(tool_name, arguments)

    assert cached_results == results
    assert mcp_cache.get_stats()["hits"] == 1
    assert mcp_cache.get_stats()["misses"] == 0
    assert mcp_cache.get_stats()["size"] == 1

def test_cache_miss(mcp_cache):
    tool_name = "non-existent-tool"
    arguments = {"param": "value"}

    cached_results = mcp_cache.get(tool_name, arguments)
    assert cached_results is None
    assert mcp_cache.get_stats()["hits"] == 0
    assert mcp_cache.get_stats()["misses"] == 1
    assert mcp_cache.get_stats()["size"] == 0

def test_cache_eviction_lru(mcp_cache):
    tool1_name = "tool1"
    tool1_args = {"p": 1}
    tool1_results = [types.TextContent(type="text", text="Result1")]

    tool2_name = "tool2"
    tool2_args = {"p": 2}
    tool2_results = [types.TextContent(type="text", text="Result2")]

    tool3_name = "tool3"
    tool3_args = {"p": 3}
    tool3_results = [types.TextContent(type="text", text="Result3")]

    mcp_cache.put(tool1_name, tool1_args, tool1_results)
    mcp_cache.put(tool2_name, tool2_args, tool2_results)
    assert mcp_cache.get_stats()["size"] == 2

    # Access tool1 to make it recently used
    mcp_cache.get(tool1_name, tool1_args)

    # Add tool3, should evict tool2 (LRU)
    mcp_cache.put(tool3_name, tool3_args, tool3_results)
    assert mcp_cache.get_stats()["size"] == 2
    assert mcp_cache.get(tool1_name, tool1_args) == tool1_results
    assert mcp_cache.get(tool2_name, tool2_args) is None # tool2 should be evicted
    assert mcp_cache.get(tool3_name, tool3_args) == tool3_results
    assert mcp_cache.get_stats()["evictions"] == 1

def test_cache_ttl(mcp_cache):
    tool_name = "ttl-tool"
    arguments = {"data": "temp"}
    results = [types.TextContent(type="text", text="TTL Result")]

    mcp_cache.put(tool_name, arguments, results)
    assert mcp_cache.get(tool_name, arguments) == results

    time.sleep(1.1) # Wait for TTL to expire

    cached_results = mcp_cache.get(tool_name, arguments)
    assert cached_results is None
    assert mcp_cache.get_stats()["misses"] == 1 # Miss due to expiration

def test_cache_clear(mcp_cache):
    tool_name = "clear-tool"
    arguments = {"x": 1}
    results = [types.TextContent(type="text", text="Clear Result")]

    mcp_cache.put(tool_name, arguments, results)
    assert mcp_cache.get_stats()["size"] == 1

    mcp_cache.clear()
    assert mcp_cache.get_stats()["size"] == 0
    assert mcp_cache.get_stats()["hits"] == 0
    assert mcp_cache.get_stats()["misses"] == 0
    assert mcp_cache.get_stats()["evictions"] == 0
    assert mcp_cache.get_stats()["total_requests"] == 0

def test_cache_stats(mcp_cache):
    tool1_name = "stat-tool1"
    tool1_args = {"a": 1}
    tool1_results = [types.TextContent(type="text", text="Stat1")]

    tool2_name = "stat-tool2"
    tool2_args = {"b": 2}
    tool2_results = [types.TextContent(type="text", text="Stat2")]

    mcp_cache.put(tool1_name, tool1_args, tool1_results)
    mcp_cache.get(tool1_name, tool1_args) # Hit
    mcp_cache.get(tool2_name, tool2_args) # Miss
    mcp_cache.put(tool2_name, tool2_args, tool2_results)
    mcp_cache.get(tool2_name, tool2_args) # Hit

    stats = mcp_cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["evictions"] == 0
    assert stats["total_requests"] == 3
    assert stats["hit_rate"] == pytest.approx(2/3)
