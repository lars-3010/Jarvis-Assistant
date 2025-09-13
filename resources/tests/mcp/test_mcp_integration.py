import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import mcp.types as types
import pytest

from jarvis.mcp.cache import MCPToolCache
from jarvis.mcp.plugins.tools.health_status import HealthStatusPlugin
from jarvis.mcp.plugins.tools.search_combined import SearchCombinedPlugin
from jarvis.mcp.plugins.tools.search_graph import SearchGraphPlugin
from jarvis.utils.config import JarvisSettings
from resources.tests.helpers.container import make_container


@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    settings = JarvisSettings()
    settings.graph_enabled = True  # Default to enabled for most tests
    settings.mcp_cache_size = 2
    settings.mcp_cache_ttl = 1
    settings.use_dependency_injection = False
    return settings


@pytest.mark.anyio
async def test_search_graph_fallback(mock_settings):
    """Graph search falls back to semantic when graph DB unhealthy."""

    graph_db = MagicMock()
    graph_db.is_healthy = False

    semantic_item = MagicMock()
    semantic_item.path = Path("some/note.md")
    semantic_item.vault_name = "default"
    semantic_item.similarity_score = -1.2
    searcher = MagicMock()
    searcher.search.return_value = [semantic_item]

    plugin = SearchGraphPlugin(container=make_container(graph_db=graph_db, searcher=searcher))
    result = await plugin.execute({"query_note_path": "some/note.md", "depth": 1})

    assert len(result) == 1 and result[0].type == "text"
    payload = json.loads(result[0].text)
    assert payload.get("mode") == "fallback_semantic"
    assert isinstance(payload.get("results"), list)


@pytest.mark.anyio
async def test_get_health_status(mock_settings):
    """Test the get-health-status MCP tool via plugin."""

    hc = MagicMock()
    hc.get_overall_health.return_value = {
        "overall_status": "healthy",
        "services": {"Neo4j": True, "VectorDB": True, "Vault": True},
        "databases": {"duckdb": True},
        "vaults": {"default": True},
        "metrics": {"system_uptime": 123},
    }

    plugin = HealthStatusPlugin(container=make_container(health_checker=hc))
    result = await plugin.execute({})

    assert len(result) == 1 and result[0].type == "text"
    health_data = json.loads(result[0].text)
    assert health_data["overall_status"] == "healthy"
    assert len(health_data["services"]) == 3
    hc.get_overall_health.assert_called_once()


@pytest.mark.anyio
async def test_search_combined(mock_settings):
    """Test the search-combined MCP tool (structured JSON) via plugin."""

    sem1 = MagicMock(path=Path("sem1.md"), vault_name="vault1", similarity_score=-1.0)
    common_sem = MagicMock(path=Path("common.md"), vault_name="vault1", similarity_score=-2.0)
    searcher = MagicMock()
    searcher.search.return_value = [sem1, common_sem]

    mock_vault_reader = MagicMock()
    mock_vault_reader.search_vault.return_value = [
        {"path": "key1.md", "vault_name": "vault1", "match_type": "name", "size": 100},
        {"path": "common.md", "vault_name": "vault1", "match_type": "content", "size": 200, "content_preview": "preview"},
    ]

    plugin = SearchCombinedPlugin(container=make_container(searcher=searcher, vault_reader=mock_vault_reader))
    result = await plugin.execute({"query": "test query", "limit": 5, "vault": "default", "search_content": True})

    assert len(result) == 1 and result[0].type == "text"
    payload = json.loads(result[0].text)
    assert payload["query"] == "test query"
    assert payload["total_results"] >= 2
    types_present = {item.get("type") for item in payload["results"]}
    assert "semantic" in types_present or "both" in types_present
    assert "keyword" in types_present or "both" in types_present


def test_mcp_cache_behavior_unit():
    """Unit test MCPToolCache behavior independent of server."""
    cache = MCPToolCache(max_size=1, ttl_seconds=1)
    key_args = {"param": "value"}

    # Miss on empty cache
    assert cache.get("tool", key_args) is None
    stats = cache.get_stats()
    assert stats["misses"] == 1 and stats["hits"] == 0

    # Put and hit
    payload = [types.TextContent(type="text", text="Unique result 1")]
    cache.put("tool", key_args, payload)
    assert cache.get("tool", key_args) == payload
    stats = cache.get_stats()
    assert stats["hits"] == 1

    # TTL expiry
    time.sleep(1.2)
    assert cache.get("tool", key_args) is None
