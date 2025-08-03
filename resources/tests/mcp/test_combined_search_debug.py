"""
Test module for combined search MCP functionality with debug logging.

This test module provides debugging capabilities for the combined search MCP tool,
including comprehensive logging and error analysis.
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import mcp.types as types
from dotenv import load_dotenv

from jarvis.mcp.server import _handle_search_combined, MCPServerContext
from jarvis.utils.config import JarvisSettings

# Load environment variables for testing
load_dotenv()


@pytest.fixture
def mock_settings():
    """Fixture for test settings."""
    settings = JarvisSettings()
    settings.graph_enabled = False  # Disable graph for faster testing
    settings.mcp_cache_size = 2
    settings.mcp_cache_ttl = 1
    settings.metrics_enabled = False
    return settings


@pytest.fixture
def test_vault_path():
    """Fixture for test vault path."""
    vault_path = os.getenv('JARVIS_VAULT_PATH')
    if not vault_path:
        pytest.skip("JARVIS_VAULT_PATH environment variable not set")
    return Path(vault_path)


@pytest.fixture
def test_db_path():
    """Fixture for test database path."""
    db_path = os.getenv('JARVIS_DATABASE_PATH', 'data/jarvis-vector.duckdb')
    return Path(db_path)


@pytest.mark.anyio
async def test_combined_search_with_logging(mock_settings, test_vault_path, test_db_path):
    """Test combined search functionality with comprehensive logging."""
    
    # Skip if test vault doesn't exist
    if not test_vault_path.exists():
        pytest.skip(f"Test vault not found: {test_vault_path}")
    
    if not test_db_path.exists():
        pytest.skip(f"Test database not found: {test_db_path}")
    
    vaults = {"default": test_vault_path}
    
    try:
        # Create context
        context = MCPServerContext(vaults, test_db_path, mock_settings)
        
        # Test arguments
        test_args = {
            "query": "test search",
            "limit": 3,
            "search_content": True
        }
        
        # Call the function directly
        results = await _handle_search_combined(context, test_args)
        
        # Assertions
        assert len(results) == 1  # Should return one TextContent result
        assert isinstance(results[0], types.TextContent)
        assert "combined results" in results[0].text.lower()
        
    finally:
        if 'context' in locals():
            context.close()


@pytest.mark.anyio 
async def test_combined_search_empty_query(mock_settings, test_vault_path, test_db_path):
    """Test combined search with empty query."""
    
    if not test_vault_path.exists() or not test_db_path.exists():
        pytest.skip("Test files not found")
    
    vaults = {"default": test_vault_path}
    
    try:
        context = MCPServerContext(vaults, test_db_path, mock_settings)
        
        # Test with empty query
        test_args = {"query": "", "limit": 3}
        
        results = await _handle_search_combined(context, test_args)
        
        # Should return error for empty query
        assert len(results) == 1
        assert "Error: Query parameter is required" in results[0].text
        
    finally:
        if 'context' in locals():
            context.close()


async def debug_combined_search():
    """Standalone debug function for manual testing."""
    print("🧪 Debug testing combined search functionality...")
    
    vault_path_env = os.getenv('JARVIS_VAULT_PATH')
    if not vault_path_env:
        print("❌ JARVIS_VAULT_PATH environment variable not set")
        return
        
    db_path_env = os.getenv('JARVIS_DATABASE_PATH', 'data/jarvis-vector.duckdb')
    
    vault_path = Path(vault_path_env)
    db_path = Path(db_path_env)
    
    if not vault_path.exists():
        print(f"❌ Vault not found: {vault_path}")
        return
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    vaults = {"default": vault_path}
    settings = JarvisSettings()
    settings.graph_enabled = False
    settings.metrics_enabled = False
    
    try:
        print("🔧 Creating MCP server context...")
        context = MCPServerContext(vaults, db_path, settings)
        
        test_args = {
            "query": "machine learning",
            "limit": 5,
            "search_content": True
        }
        
        print(f"📝 Test arguments: {test_args}")
        print("🚀 Calling _handle_search_combined...")
        
        results = await _handle_search_combined(context, test_args)
        
        print("✅ Search completed!")
        print(f"📋 Results: {len(results)} items")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'context' in locals():
            context.close()


if __name__ == "__main__":
    # Allow running as standalone debug script
    asyncio.run(debug_combined_search())