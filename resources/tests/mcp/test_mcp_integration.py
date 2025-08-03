import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from jarvis.mcp.server import create_mcp_server
from jarvis.mcp.cache import MCPToolCache
from jarvis.utils.config import JarvisSettings
import mcp.types as types

@pytest.fixture
def mock_settings():
    """Fixture for mock settings."""
    settings = JarvisSettings()
    settings.graph_enabled = True  # Default to enabled for most tests
    settings.mcp_cache_size = 2
    settings.mcp_cache_ttl = 1
    return settings

@pytest.mark.anyio
async def test_search_graph_fallback(mock_settings):
    """Test that search-graph falls back to semantic search when Neo4j is disabled."""
    mock_settings.graph_enabled = False # Explicitly disable for this test
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        # Mock the context instance returned by the constructor
        mock_context = mock_context_class.return_value
        mock_context.graph_database.is_healthy = False
        mock_context.settings = mock_settings # Ensure settings are passed to context

        # Mock the semantic search handler
        with patch('jarvis.mcp.server._handle_semantic_search', new_callable=AsyncMock) as mock_semantic_search:
            mock_semantic_search.return_value = [types.TextContent(type="text", text="Semantic search results")]

            # Create the server (will use the mocked context)
            server = create_mcp_server({}, MagicMock(), mock_settings)

            # Call the search-graph tool
            result = await server.call_tool(name='search-graph', arguments={'query_note_path': 'some/note.md'})

            # Assertions
            assert len(result) == 2
            assert "Graph search is unavailable" in result[0].text
            assert "Semantic search results" in result[1].text
            mock_semantic_search.assert_called_once_with(
                mock_context,
                {'query': 'some/note.md', 'limit': 10}
            )

@pytest.mark.anyio
async def test_get_health_status(mock_settings):
    """Test the get-health-status MCP tool."""
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings
        mock_context.health_checker.get_overall_health.return_value = {
            "overall_status": "HEALTHY",
            "services": [
                {"service": "Neo4j", "status": "HEALTHY"},
                {"service": "VectorDB", "status": "HEALTHY"},
                {"service": "Vault", "status": "HEALTHY"}
            ]
        }

        server = create_mcp_server({}, MagicMock(), mock_settings)

        result = await server.call_tool(name='get-health-status', arguments={})

        assert len(result) == 1
        assert result[0].type == "text"
        health_data = json.loads(result[0].text)
        assert health_data["overall_status"] == "HEALTHY"
        assert len(health_data["services"]) == 3
        mock_context.health_checker.get_overall_health.assert_called_once()

@pytest.mark.anyio
async def test_search_combined(mock_settings):
    """Test the search-combined MCP tool."""
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings

        # Mock semantic search results
        mock_semantic_search_results = [
            MagicMock(spec=types.TextContent, path=Path("sem1.md"), vault_name="vault1", similarity_score=0.9),
            MagicMock(spec=types.TextContent, path=Path("common.md"), vault_name="vault1", similarity_score=0.8),
        ]
        mock_context.searcher.search.return_value = mock_semantic_search_results

        # Mock keyword search results
        mock_keyword_search_results = [
            {"path": "key1.md", "vault_name": "vault1", "match_type": "name", "size": 100},
            {"path": "common.md", "vault_name": "vault1", "match_type": "content", "size": 200, "content_preview": "preview"},
        ]
        # Mock the vault reader's search_vault method
        mock_vault_reader = MagicMock()
        mock_vault_reader.search_vault.return_value = mock_keyword_search_results
        mock_context.vault_readers = {"default": mock_vault_reader}
        mock_context.vaults = {"default": Path("/tmp/vault")}

        # Mock the ranker's merge_and_rank method
        mock_context.ranker.merge_and_rank.return_value = [
            mock_semantic_search_results[0], # sem1
            mock_semantic_search_results[1], # common (semantic takes precedence)
            mock_keyword_search_results[0] # key1
        ]

        server = create_mcp_server({}, MagicMock(), mock_settings)

        result = await server.call_tool(name='search-combined', arguments={'query': 'test query', 'limit': 5, 'vault': 'default', 'search_content': True})

        assert len(result) == 1
        assert result[0].type == "text"
        response_text = result[0].text

        assert "Found 3 combined results for 'test query':" in response_text
        assert "[SEMANTIC] sem1.md" in response_text
        assert "[SEMANTIC] common.md" in response_text
        assert "[KEYWORD] key1.md" in response_text

        mock_context.searcher.search.assert_called_once_with(
            query='test query', limit=5, vault_name='default'
        )
        mock_vault_reader.search_vault.assert_called_once_with(
            query='test query', limit=5, vault='default', search_content=True
        )
        mock_context.ranker.merge_and_rank.assert_called_once()

@pytest.mark.anyio
async def test_mcp_cache_behavior(mock_settings):
    """Test MCP tool caching behavior."""
    mock_settings.mcp_cache_size = 1
    mock_settings.mcp_cache_ttl = 1

    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings
        mock_context.mcp_cache = MCPToolCache(mock_settings.mcp_cache_size, mock_settings.mcp_cache_ttl)

        # Mock a tool handler that returns a unique result each time
        mock_tool_handler = AsyncMock(return_value=[types.TextContent(type="text", text="Unique result 1")])
        
        # Patch the internal tool dispatch to use our mock handler
        with patch.dict('jarvis.mcp.server.Server.request_handlers', {
            types.CallToolRequest: MagicMock(side_effect=lambda req: mock_tool_handler(req.params.name, req.params.arguments))
        }):
            server = create_mcp_server({}, MagicMock(), mock_settings)

            # First call - should be a cache miss
            result1 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result1[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_called_once() # Tool handler called

            # Second call - should be a cache hit
            mock_tool_handler.reset_mock() # Reset mock to check if it's called again
            result2 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result2[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 1
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_not_called() # Tool handler NOT called

            # Clear cache - next call should be a miss
            mock_context.clear_cache()
            mock_tool_handler.reset_mock()
            result3 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result3[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_called_once() # Tool handler called again

            # Test TTL expiration
            mock_tool_handler.reset_mock()
            result4 = await server.call_tool('test-ttl-tool', {"param": "value"})
            assert "Unique result 1" in result4[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 2 # Miss for TTL tool
            mock_tool_handler.assert_called_once()

            time.sleep(1.1) # Wait for TTL to expire

            mock_tool_handler.reset_mock()
            result5 = await server.call_tool('test-ttl-tool', {"param": "value"})
            assert "Unique result 1" in result5[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 3 # Miss due to expiration
            mock_tool_handler.assert_called_once()