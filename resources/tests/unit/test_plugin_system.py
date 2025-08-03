"""
Unit tests for the MCP Plugin System.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from mcp import types
from jarvis.mcp.plugins.base import (
    MCPToolPlugin, PluginMetadata, SearchPlugin, GraphPlugin, 
    VaultPlugin, UtilityPlugin
)
from jarvis.mcp.plugins.registry import PluginRegistry, get_plugin_registry, reset_plugin_registry
from jarvis.mcp.plugins.discovery import PluginDiscovery
from jarvis.mcp.plugins.tools import get_builtin_plugins
from jarvis.core.interfaces import IVectorSearcher, IGraphDatabase, IVaultReader
from jarvis.utils.errors import PluginError


class MockPlugin(MCPToolPlugin):
    """Mock plugin for testing."""
    
    @property
    def name(self) -> str:
        return "mock-plugin"
    
    @property
    def description(self) -> str:
        return "A mock plugin for testing"
    
    def get_tool_definition(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema={"type": "object", "properties": {}}
        )
    
    async def execute(self, arguments):
        return [types.TextContent(type="text", text="Mock response")]


class TestPluginMetadata:
    """Test PluginMetadata class."""
    
    def test_metadata_creation(self):
        """Test plugin metadata creation."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            tags=["test", "mock"]
        )
        
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "mock"]
    
    def test_metadata_to_dict(self):
        """Test metadata dictionary conversion."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin"
        )
        
        data = metadata.to_dict()
        
        assert data["name"] == "test-plugin"
        assert data["version"] == "1.0.0"
        assert data["description"] == "Test plugin"
        assert data["dependencies"] == []
        assert data["tags"] == []


class TestMCPToolPlugin:
    """Test MCPToolPlugin base class."""
    
    def test_plugin_creation(self):
        """Test plugin creation."""
        plugin = MockPlugin()
        
        assert plugin.name == "mock-plugin"
        assert plugin.description == "A mock plugin for testing"
        assert plugin.version == "1.0.0"
        assert not plugin.is_validated
    
    def test_plugin_metadata(self):
        """Test plugin metadata generation."""
        plugin = MockPlugin()
        metadata = plugin.metadata
        
        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "mock-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "A mock plugin for testing"
    
    def test_plugin_validation(self):
        """Test plugin validation."""
        plugin = MockPlugin()
        
        assert plugin.validate()
        assert plugin.is_validated
    
    def test_plugin_invalid_name(self):
        """Test plugin with invalid name."""
        class InvalidPlugin(MCPToolPlugin):
            @property
            def name(self):
                return ""  # Invalid empty name
            
            @property
            def description(self):
                return "Invalid plugin"
            
            def get_tool_definition(self):
                return types.Tool(name="", description="", inputSchema={})
            
            async def execute(self, arguments):
                return []
        
        plugin = InvalidPlugin()
        assert not plugin.validate()
    
    def test_plugin_required_services(self):
        """Test plugin required services."""
        plugin = MockPlugin()
        services = plugin.get_required_services()
        
        assert isinstance(services, list)
        assert len(services) == 0  # MockPlugin has no required services
    
    def test_plugin_dependency_check_no_container(self):
        """Test dependency check without container."""
        plugin = MockPlugin()
        
        # Should return True when no container (no dependency checking)
        assert plugin.check_dependencies()
    
    def test_plugin_dependency_check_with_container(self):
        """Test dependency check with container."""
        mock_container = Mock()
        mock_container.get.return_value = Mock()  # Mock service available
        
        plugin = MockPlugin(container=mock_container)
        
        # Should return True when all dependencies are satisfied
        assert plugin.check_dependencies()


class TestPluginSpecializations:
    """Test specialized plugin base classes."""
    
    def test_search_plugin(self):
        """Test SearchPlugin specialization."""
        class TestSearchPlugin(SearchPlugin):
            @property
            def name(self):
                return "test-search"
            
            @property
            def description(self):
                return "Test search plugin"
            
            def get_tool_definition(self):
                return types.Tool(name=self.name, description=self.description, inputSchema={})
            
            async def execute(self, arguments):
                return []
        
        plugin = TestSearchPlugin()
        
        assert "search" in plugin.tags
        assert "query" in plugin.tags
        assert IVectorSearcher in plugin.get_required_services()
    
    def test_graph_plugin(self):
        """Test GraphPlugin specialization."""
        class TestGraphPlugin(GraphPlugin):
            @property
            def name(self):
                return "test-graph"
            
            @property
            def description(self):
                return "Test graph plugin"
            
            def get_tool_definition(self):
                return types.Tool(name=self.name, description=self.description, inputSchema={})
            
            async def execute(self, arguments):
                return []
        
        plugin = TestGraphPlugin()
        
        assert "graph" in plugin.tags
        assert "relationships" in plugin.tags
        assert IGraphDatabase in plugin.get_required_services()
    
    def test_vault_plugin(self):
        """Test VaultPlugin specialization."""
        class TestVaultPlugin(VaultPlugin):
            @property
            def name(self):
                return "test-vault"
            
            @property
            def description(self):
                return "Test vault plugin"
            
            def get_tool_definition(self):
                return types.Tool(name=self.name, description=self.description, inputSchema={})
            
            async def execute(self, arguments):
                return []
        
        plugin = TestVaultPlugin()
        
        assert "vault" in plugin.tags
        assert "files" in plugin.tags
        assert IVaultReader in plugin.get_required_services()


class TestPluginRegistry:
    """Test PluginRegistry functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = PluginRegistry()
    
    def test_registry_creation(self):
        """Test registry creation."""
        assert len(self.registry.list_plugins()) == 0
    
    def test_register_plugin_class(self):
        """Test plugin class registration."""
        success = self.registry.register_plugin_class(MockPlugin)
        
        assert success
        assert "mock-plugin" in self.registry.list_plugins()
    
    def test_register_plugin_instance(self):
        """Test plugin instance registration."""
        plugin = MockPlugin()
        success = self.registry.register_plugin_instance(plugin)
        
        assert success
        assert "mock-plugin" in self.registry.list_plugins(loaded_only=True)
    
    def test_load_plugin(self):
        """Test plugin loading."""
        self.registry.register_plugin_class(MockPlugin)
        plugin = self.registry.load_plugin("mock-plugin")
        
        assert plugin is not None
        assert isinstance(plugin, MockPlugin)
        assert plugin.name == "mock-plugin"
    
    def test_get_plugin(self):
        """Test plugin retrieval."""
        self.registry.register_plugin_class(MockPlugin)
        plugin = self.registry.get_plugin("mock-plugin")
        
        assert plugin is not None
        assert isinstance(plugin, MockPlugin)
    
    def test_unregister_plugin(self):
        """Test plugin unregistration."""
        self.registry.register_plugin_class(MockPlugin)
        self.registry.load_plugin("mock-plugin")
        
        success = self.registry.unregister_plugin("mock-plugin")
        
        assert success
        assert "mock-plugin" not in self.registry.list_plugins()
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        self.registry.register_plugin_instance(MockPlugin())
        tools = self.registry.get_tool_definitions()
        
        assert len(tools) == 1
        assert isinstance(tools[0], types.Tool)
        assert tools[0].name == "mock-plugin"
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test tool execution through registry."""
        self.registry.register_plugin_instance(MockPlugin())
        result = await self.registry.execute_tool("mock-plugin", {})
        
        assert len(result) == 1
        assert isinstance(result[0], types.TextContent)
        assert result[0].text == "Mock response"
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        with pytest.raises(PluginError, match="not found in registry"):
            await self.registry.execute_tool("unknown-tool", {})
    
    def test_validate_all_plugins(self):
        """Test validating all plugins."""
        self.registry.register_plugin_class(MockPlugin)
        self.registry.register_plugin_instance(MockPlugin())
        
        results = self.registry.validate_all_plugins()
        
        assert len(results) == 1  # Only one unique plugin
        assert results["mock-plugin"] is True
    
    def test_get_plugins_by_tag(self):
        """Test getting plugins by tag."""
        # Create a plugin with specific tags
        class TaggedPlugin(MockPlugin):
            @property
            def name(self):
                return "tagged-plugin"
            
            @property
            def tags(self):
                return ["test", "example"]
        
        self.registry.register_plugin_instance(TaggedPlugin())
        
        test_plugins = self.registry.get_plugins_by_tag("test")
        example_plugins = self.registry.get_plugins_by_tag("example")
        empty_plugins = self.registry.get_plugins_by_tag("nonexistent")
        
        assert len(test_plugins) == 1
        assert len(example_plugins) == 1
        assert len(empty_plugins) == 0
        assert test_plugins[0].name == "tagged-plugin"
    
    def test_get_registry_info(self):
        """Test getting registry information."""
        self.registry.register_plugin_class(MockPlugin)
        self.registry.load_plugin("mock-plugin")
        
        info = self.registry.get_registry_info()
        
        assert info["total_registered_classes"] == 1
        assert info["total_loaded_instances"] == 1
        assert "mock-plugin" in info["available_plugins"]
        assert "mock-plugin" in info["loaded_plugins"]
    
    def test_load_all_plugins(self):
        """Test loading all registered plugins."""
        self.registry.register_plugin_class(MockPlugin)
        
        results = self.registry.load_all_plugins()
        
        assert len(results) == 1
        assert results["mock-plugin"] is True
        assert len(self.registry.list_plugins(loaded_only=True)) == 1
    
    def test_unload_all_plugins(self):
        """Test unloading all plugins."""
        self.registry.register_plugin_instance(MockPlugin())
        assert len(self.registry.list_plugins(loaded_only=True)) == 1
        
        self.registry.unload_all_plugins()
        
        assert len(self.registry.list_plugins(loaded_only=True)) == 0
    
    def test_reload_plugin(self):
        """Test plugin reloading."""
        self.registry.register_plugin_class(MockPlugin)
        self.registry.load_plugin("mock-plugin")
        
        success = self.registry.reload_plugin("mock-plugin")
        
        assert success
        assert "mock-plugin" in self.registry.list_plugins(loaded_only=True)


class TestPluginDiscovery:
    """Test PluginDiscovery functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = PluginRegistry()
        self.discovery = PluginDiscovery(self.registry)
    
    def test_discovery_creation(self):
        """Test discovery system creation."""
        assert self.discovery.registry == self.registry
        assert len(self.discovery._discovery_paths) == 0
    
    def test_add_discovery_path(self):
        """Test adding discovery paths."""
        test_path = Path(".")  # Current directory exists
        self.discovery.add_discovery_path(test_path)
        
        assert test_path in self.discovery._discovery_paths
    
    def test_add_nonexistent_path(self):
        """Test adding nonexistent discovery path."""
        test_path = Path("/nonexistent/path")
        self.discovery.add_discovery_path(test_path)
        
        # Should not be added since it doesn't exist
        assert test_path not in self.discovery._discovery_paths
    
    def test_discover_builtin_plugins(self):
        """Test built-in plugin discovery."""
        try:
            plugins = self.discovery.discover_builtin_plugins()
            # Should return a list (may be empty if tools package not found)
            assert isinstance(plugins, list)
        except ImportError:
            # Expected if tools package not properly set up
            pass
    
    def test_extract_plugins_from_module(self):
        """Test extracting plugins from a module."""
        # Create a mock module with plugin classes
        import types as module_types
        mock_module = module_types.ModuleType("mock_module")
        mock_module.__name__ = "mock_module"
        
        # Add the MockPlugin class to the module
        MockPlugin.__module__ = "mock_module"
        setattr(mock_module, "MockPlugin", MockPlugin)
        
        plugins = self.discovery._extract_plugins_from_module(mock_module)
        
        assert len(plugins) == 1
        assert plugins[0] == MockPlugin
    
    @patch('importlib.import_module')
    def test_discover_from_package(self, mock_import):
        """Test package discovery."""
        # Mock a package with plugins
        mock_package = Mock()
        mock_package.__path__ = ["/mock/path"]
        mock_import.return_value = mock_package
        
        with patch('pkgutil.iter_modules') as mock_iter:
            mock_iter.return_value = [
                (None, "plugin_module", False)
            ]
            
            with patch.object(self.discovery, '_extract_plugins_from_module') as mock_extract:
                mock_extract.return_value = [MockPlugin]
                
                plugins = self.discovery.discover_from_package("test_package")
                
                assert len(plugins) == 1
                assert plugins[0] == MockPlugin


class TestBuiltinPlugins:
    """Test built-in plugin implementations."""
    
    def test_get_builtin_plugins(self):
        """Test getting built-in plugins."""
        try:
            plugins = get_builtin_plugins()
            
            assert isinstance(plugins, list)
            assert len(plugins) > 0  # Should have at least some plugins
            
            # Check that all are plugin classes
            for plugin_class in plugins:
                assert issubclass(plugin_class, MCPToolPlugin)
                
        except ImportError:
            pytest.skip("Built-in plugins package not available")


class TestGlobalRegistry:
    """Test global registry functionality."""
    
    def test_get_global_registry(self):
        """Test getting global registry."""
        # Reset first
        reset_plugin_registry()
        
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()
        
        # Should be the same instance
        assert registry1 is registry2
    
    def test_reset_global_registry(self):
        """Test resetting global registry."""
        registry1 = get_plugin_registry()
        reset_plugin_registry()
        registry2 = get_plugin_registry()
        
        # Should be different instances after reset
        assert registry1 is not registry2


if __name__ == "__main__":
    pytest.main([__file__])