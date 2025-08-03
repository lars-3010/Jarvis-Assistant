import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from jarvis.mcp.server import MCPServerContext, _handle_semantic_search, _handle_search_graph
from jarvis.utils.config import JarvisSettings

async def manual_test_search_graph_fallback():
    print("\n--- Starting Manual Fallback Test ---")

    # 1. Mock settings to disable graph
    settings = JarvisSettings()
    settings.graph_enabled = False

    # 2. Mock MCPServerContext and its graph_database property
    mock_context = MagicMock(spec=MCPServerContext)
    mock_context.settings = settings
    mock_context.graph_database = MagicMock()
    mock_context.graph_database.is_healthy = False

    # 3. Mock the semantic search handler to return a predictable result
    with patch('jarvis.mcp.server._handle_semantic_search', new_callable=AsyncMock) as mock_semantic_search:
        mock_semantic_search.return_value = [MagicMock(type='text', text='**Semantic search results for: some/note.md**')]

        # 4. Call the _handle_search_graph function directly
        print("Calling _handle_search_graph with Neo4j disabled...")
        result = await _handle_search_graph(
            mock_context,
            {'query_note_path': 'some/note.md'}
        )

        # 5. Assertions
        print("\n--- Verifying Results ---")
        print(f"Number of results: {len(result)}")
        for item in result:
            print(f"  Type: {item.type}, Text: {item.text}")

        assert len(result) == 2
        assert "Graph search is unavailable. Falling back to semantic search." in result[0].text
        assert "**Semantic search results for: some/note.md**" in result[1].text
        mock_semantic_search.assert_called_once_with(
            mock_context,
            {'query': 'some/note.md', 'limit': 10}
        )
        print("\n--- Manual Fallback Test PASSED ---")

if __name__ == "__main__":
    asyncio.run(manual_test_search_graph_fallback())