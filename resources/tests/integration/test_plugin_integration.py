"""
Integration tests for the MCP Plugin System.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path

from mcp import types
from jarvis.mcp.plugins.base import MCPToolPlugin
from jarvis.mcp.plugins.registry import PluginRegistry
from jarvis.mcp.plugins.discovery import PluginDiscovery
from jarvis.mcp.plugins.tools.semantic_search import SemanticSearchPlugin
from jarvis.mcp.plugins.tools.read_note import ReadNotePlugin
from jarvis.mcp.plugins.tools.list_vaults import ListVaultsPlugin
from jarvis.core.interfaces import IVectorSearcher, IVaultReader, IMetrics
from jarvis.models.document import SearchResult
from jarvis.utils.errors import PluginError


class MockVectorSearcher:
    """Mock vector searcher for testing."""
    
    def search(self, query, top_k=10, vault_name=None, similarity_threshold=None):
        return [
            SearchResult(
                path=Path("test/note1.md"),
                vault_name="test_vault",
                similarity_score=0.85,
                content_preview="Test content preview"
            ),
            SearchResult(
                path=Path("test/note2.md"),
                vault_name="test_vault",
                similarity_score=0.75,
                content_preview="Another test content"
            )
        ]
    
    def get_vault_stats(self):
        return {
            "test_vault": {
                "note_count": 10,
                "latest_modified": 1640995200.0,
                "earliest_modified": 1640908800.0
            }
        }
    
    def validate_vaults(self):
        return {"test_vault": True}
    
    def get_model_info(self):
        return {
            "encoder_info": {
                "model_name": "paraphrase-MiniLM-L6-v2",
                "device": "cpu"
            },
            "database_note_count": 10
        }
    
    def get_search_stats(self):
        return {
            "total_searches": 42,
            "avg_response_time_ms": 125.5
        }


class MockVaultReader:
    """Mock vault reader for testing."""
    
    def read_file(self, path):
        content = f"# {path}\n\nThis is the content of {path}."
        metadata = {
            "path": path,
            "size": len(content),
            "modified_formatted": "2024-01-01T10:00:00"
        }
        return content, metadata
    
    def search_vault(self, query, search_content=False, limit=20):
        return [
            {
                "path": "test/note1.md",
                "match_type": "name",
                "size": 150,
                "content_preview": "Test content preview"
            },
            {
                "path": "test/note2.md", 
                "match_type": "content",
                "size": 200,
                "content_preview": "Another test content"
            }
        ]


class MockMetrics:
    """Mock metrics service for testing."""
    
    def __init__(self):
        self.metrics = {
            "mcp_tool_search_semantic": 15,
            "mcp_tool_read_note": 8,
            "system_uptime": 3600,
            "service_database_connections": 2
        }
    
    def get_metrics(self):
        return self.metrics.copy()
    
    def record_counter(self, name, value=1, tags=None):
        if name in self.metrics:
            self.metrics[name] += value
        else:
            self.metrics[name] = value
    
    def time_operation(self, operation_name):
        # Return a context manager mock
        return Mock(__enter__=Mock(), __exit__=Mock())
    
    def reset_metrics(self):
        self.metrics.clear()


class MockServiceContainer:
    """Mock service container for testing."""
    
    def __init__(self):
        self.services = {
            IVectorSearcher: MockVectorSearcher(),
            IVaultReader: MockVaultReader(),
            IMetrics: MockMetrics()
        }
    
    def get(self, service_type):
        return self.services.get(service_type)


@pytest.mark.integration
class TestPluginSystemIntegration:
    """Integration tests for the complete plugin system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.container = MockServiceContainer()
        self.registry = PluginRegistry(self.container)
        self.discovery = PluginDiscovery(self.registry)
    
    def test_plugin_registration_and_execution(self):
        """Test complete plugin lifecycle: registration, discovery, and execution."""
        # Register semantic search plugin
        success = self.registry.register_plugin_class(SemanticSearchPlugin)
        assert success
        
        # Load the plugin
        plugin = self.registry.load_plugin("search-semantic")
        assert plugin is not None
        assert isinstance(plugin, SemanticSearchPlugin)
        
        # Check tool definition
        tool_def = plugin.get_tool_definition()
        assert isinstance(tool_def, types.Tool)
        assert tool_def.name == "search-semantic"
    
    @pytest.mark.asyncio
    async def test_semantic_search_plugin_execution(self):
        """Test semantic search plugin execution."""
        # Register and load plugin
        self.registry.register_plugin_class(SemanticSearchPlugin)
        
        # Execute search
        arguments = {
            "query": "test query",
            "limit": 5
        }
        
        result = await self.registry.execute_tool("search-semantic", arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Found 2 results for 'test query'" in result[0].text
        assert "test/note1.md" in result[0].text
        assert "score: 0.850" in result[0].text
    
    @pytest.mark.asyncio
    async def test_read_note_plugin_execution(self):
        """Test read note plugin execution."""
        # Register and load plugin
        self.registry.register_plugin_class(ReadNotePlugin)
        
        # Execute note reading
        arguments = {
            "path": "test/sample.md"
        }
        
        result = await self.registry.execute_tool("read-note", arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "# test/sample.md" in result[0].text
        assert "This is the content of test/sample.md" in result[0].text
    
    @pytest.mark.asyncio
    async def test_list_vaults_plugin_execution(self):
        """Test list vaults plugin execution."""
        # Register and load plugin
        self.registry.register_plugin_class(ListVaultsPlugin)
        
        # Execute vault listing
        result = await self.registry.execute_tool("list-vaults", {})
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert "Available Vaults" in result[0].text
        assert "test_vault" in result[0].text
        assert "Notes:** 10" in result[0].text
    
    def test_multiple_plugin_registration(self):
        """Test registering multiple plugins."""
        plugins = [
            SemanticSearchPlugin,
            ReadNotePlugin,
            ListVaultsPlugin
        ]
        
        for plugin_class in plugins:
            success = self.registry.register_plugin_class(plugin_class)
            assert success
        
        # Check all plugins are registered
        plugin_names = self.registry.list_plugins()
        assert "search-semantic" in plugin_names
        assert "read-note" in plugin_names
        assert "list-vaults" in plugin_names
        
        # Load all plugins
        load_results = self.registry.load_all_plugins()
        assert all(load_results.values())
        
        # Get tool definitions
        tools = self.registry.get_tool_definitions()
        assert len(tools) == 3
        
        tool_names = [tool.name for tool in tools]
        assert "search-semantic" in tool_names
        assert "read-note" in tool_names
        assert "list-vaults" in tool_names
    
    @pytest.mark.asyncio
    async def test_plugin_validation_and_error_handling(self):
        """Test plugin validation and error handling."""
        # Register plugin
        self.registry.register_plugin_class(SemanticSearchPlugin)
        
        # Test missing query parameter
        result = await self.registry.execute_tool("search-semantic", {})
        
        assert len(result) == 1
        assert "Error: Query parameter is required" in result[0].text
        
        # Test invalid limit
        result = await self.registry.execute_tool("search-semantic", {
            "query": "test",
            "limit": 100  # Too high
        })
        
        assert len(result) == 1
        assert "Error: Limit must be between 1 and 50" in result[0].text
    
    def test_plugin_dependencies(self):
        """Test plugin dependency checking."""
        # Create plugin without container
        plugin = SemanticSearchPlugin()
        
        # Should fail dependency check
        assert not plugin.check_dependencies()
        
        # Create plugin with container
        plugin_with_container = SemanticSearchPlugin(self.container)
        
        # Should pass dependency check
        assert plugin_with_container.check_dependencies()
    
    def test_plugin_metadata_system(self):
        """Test plugin metadata and tagging system."""
        # Register plugins
        self.registry.register_plugin_class(SemanticSearchPlugin)
        self.registry.register_plugin_class(ReadNotePlugin)
        
        # Load plugins
        self.registry.load_all_plugins()
        
        # Test getting plugins by tag
        search_plugins = self.registry.get_plugins_by_tag("search")
        vault_plugins = self.registry.get_plugins_by_tag("vault")
        
        assert len(search_plugins) == 1
        assert search_plugins[0].name == "search-semantic"
        
        assert len(vault_plugins) == 1
        assert vault_plugins[0].name == "read-note"
        
        # Test plugin metadata
        semantic_plugin = self.registry.get_plugin("search-semantic")
        metadata = semantic_plugin.metadata
        
        assert metadata.name == "search-semantic"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Jarvis Assistant"
        assert "search" in metadata.tags
    
    def test_registry_information_system(self):
        """Test registry information and statistics."""
        # Register multiple plugins
        plugins = [SemanticSearchPlugin, ReadNotePlugin, ListVaultsPlugin]
        for plugin_class in plugins:
            self.registry.register_plugin_class(plugin_class)
        
        # Load some plugins
        self.registry.load_plugin("search-semantic")
        self.registry.load_plugin("read-note")
        
        # Get registry info
        info = self.registry.get_registry_info()
        
        assert info["total_registered_classes"] == 3
        assert info["total_loaded_instances"] == 2
        assert len(info["available_plugins"]) == 3
        assert len(info["loaded_plugins"]) == 2
        
        # Check validation status
        validation = info["validation_status"]
        assert all(validation.values())  # All should be valid
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through plugin system."""
        # Create a plugin that will fail
        class FailingPlugin(MCPToolPlugin):
            @property
            def name(self):
                return "failing-plugin"
            
            @property
            def description(self):
                return "A plugin that fails"
            
            def get_tool_definition(self):
                return types.Tool(
                    name=self.name,
                    description=self.description,
                    inputSchema={"type": "object", "properties": {}}
                )
            
            async def execute(self, arguments):
                raise Exception("Plugin execution failed")
        
        # Register and try to execute
        self.registry.register_plugin_instance(FailingPlugin(self.container))
        
        with pytest.raises(PluginError, match="execution failed"):
            await self.registry.execute_tool("failing-plugin", {})
    
    def test_plugin_reload_functionality(self):
        """Test plugin reloading functionality."""
        # Register and load plugin
        self.registry.register_plugin_class(SemanticSearchPlugin)
        plugin1 = self.registry.load_plugin("search-semantic")
        
        # Reload plugin
        success = self.registry.reload_plugin("search-semantic")
        assert success
        
        # Get plugin again (should be new instance)
        plugin2 = self.registry.get_plugin("search-semantic")
        
        # Should be different instances
        assert plugin1 is not plugin2
        assert plugin2.name == "search-semantic"


@pytest.mark.integration
class TestPluginDiscoveryIntegration:
    """Integration tests for plugin discovery system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.container = MockServiceContainer()
        self.registry = PluginRegistry(self.container)
        self.discovery = PluginDiscovery(self.registry)
    
    def test_auto_discovery_and_loading(self):
        """Test automatic discovery and loading of plugins."""
        # Mock built-in plugins discovery
        with pytest.mock.patch.object(self.discovery, 'discover_builtin_plugins') as mock_builtin:
            mock_builtin.return_value = [SemanticSearchPlugin, ReadNotePlugin]
            
            # Run auto-discovery
            stats = self.discovery.discover_and_load(include_builtin=True)
            
            # Check discovery stats
            discovery_stats = stats["discovery"]
            assert discovery_stats["plugins_discovered"] == 2
            assert discovery_stats["plugins_registered"] == 2
            
            # Check loading stats
            loading_stats = stats["loading"]
            assert loading_stats["plugins_loaded"] == 2
            assert loading_stats["total_plugins"] == 2
            assert loading_stats["load_success_rate"] == 100.0
            
            # Verify plugins are actually loaded
            assert len(self.registry.list_plugins(loaded_only=True)) == 2


if __name__ == "__main__":
    pytest.main([__file__])