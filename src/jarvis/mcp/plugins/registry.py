"""
Plugin Registry for MCP Tool Plugins.

This module provides the central registry for managing MCP tool plugins,
including registration, validation, and retrieval of plugins.
"""

from typing import Dict, List, Optional, Type, Set, Any
from collections import defaultdict
import asyncio
from pathlib import Path

from mcp import types
from jarvis.mcp.plugins.base import MCPToolPlugin, PluginMetadata
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import PluginError, JarvisError

logger = setup_logging(__name__)


class PluginRegistry:
    """Central registry for MCP tool plugins."""
    
    def __init__(self, container=None):
        """Initialize the plugin registry.
        
        Args:
            container: Optional service container for dependency injection
        """
        self.container = container
        self._plugins: Dict[str, MCPToolPlugin] = {}
        self._plugin_classes: Dict[str, Type[MCPToolPlugin]] = {}
        self._tags: Dict[str, Set[str]] = defaultdict(set)
        self._metadata: Dict[str, PluginMetadata] = {}
        self._loaded_plugins: Set[str] = set()
        
        logger.info("Plugin registry initialized")
    
    def register_plugin_class(self, plugin_class: Type[MCPToolPlugin]) -> bool:
        """Register a plugin class for later instantiation.
        
        Args:
            plugin_class: Plugin class to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Create temporary instance to get name and validate
            temp_instance = plugin_class(self.container)
            plugin_name = temp_instance.name
            
            if not temp_instance.validate():
                logger.error(f"Plugin class {plugin_class.__name__} failed validation")
                return False
            
            if plugin_name in self._plugin_classes:
                logger.warning(f"Plugin class {plugin_name} already registered, overwriting")
            
            self._plugin_classes[plugin_name] = plugin_class
            self._metadata[plugin_name] = temp_instance.metadata
            
            # Store tags
            for tag in temp_instance.metadata.tags or []:
                self._tags[tag].add(plugin_name)
            
            logger.info(f"Registered plugin class: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin class {plugin_class.__name__}: {e}")
            return False
    
    def register_plugin_instance(self, plugin: MCPToolPlugin) -> bool:
        """Register a plugin instance directly.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not plugin.validate():
                logger.error(f"Plugin {plugin.name} failed validation")
                return False
            
            if plugin.name in self._plugins:
                logger.warning(f"Plugin {plugin.name} already registered, overwriting")
                self.unregister_plugin(plugin.name)
            
            self._plugins[plugin.name] = plugin
            self._metadata[plugin.name] = plugin.metadata
            
            # Store tags
            for tag in plugin.metadata.tags or []:
                self._tags[tag].add(plugin.name)
            
            # Call plugin lifecycle method
            plugin.on_load()
            self._loaded_plugins.add(plugin.name)
            
            logger.info(f"Registered plugin instance: {plugin.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin.name}: {e}")
            return False
    
    def load_plugin(self, plugin_name: str) -> Optional[MCPToolPlugin]:
        """Load a plugin from its registered class.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            Loaded plugin instance or None if failed
        """
        try:
            if plugin_name in self._plugins:
                return self._plugins[plugin_name]
            
            if plugin_name not in self._plugin_classes:
                logger.error(f"Plugin class {plugin_name} not found in registry")
                return None
            
            plugin_class = self._plugin_classes[plugin_name]
            plugin_instance = plugin_class(self.container)
            
            if self.register_plugin_instance(plugin_instance):
                return plugin_instance
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return None
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                plugin.on_unload()
                del self._plugins[plugin_name]
                self._loaded_plugins.discard(plugin_name)
            
            if plugin_name in self._plugin_classes:
                del self._plugin_classes[plugin_name]
            
            if plugin_name in self._metadata:
                metadata = self._metadata[plugin_name]
                # Remove from tags
                for tag in metadata.tags or []:
                    self._tags[tag].discard(plugin_name)
                del self._metadata[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[MCPToolPlugin]:
        """Get a plugin instance by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]
        
        # Try to load if class is registered
        return self.load_plugin(plugin_name)
    
    def list_plugins(self, loaded_only: bool = False) -> List[str]:
        """List all registered plugin names.
        
        Args:
            loaded_only: If True, only return loaded plugins
            
        Returns:
            List of plugin names
        """
        if loaded_only:
            return list(self._loaded_plugins)
        
        all_plugins = set(self._plugin_classes.keys()) | set(self._plugins.keys())
        return sorted(all_plugins)
    
    def get_plugins_by_tag(self, tag: str) -> List[MCPToolPlugin]:
        """Get all plugins with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of plugin instances
        """
        plugin_names = self._tags.get(tag, set())
        plugins = []
        
        for name in plugin_names:
            plugin = self.get_plugin(name)
            if plugin:
                plugins.append(plugin)
        
        return plugins
    
    def get_tool_definitions(self) -> List[types.Tool]:
        """Get MCP tool definitions for all loaded plugins.
        
        Returns:
            List of MCP Tool definitions
        """
        tools = []
        
        for plugin_name in self._loaded_plugins:
            try:
                plugin = self._plugins[plugin_name]
                tool_def = plugin.get_tool_definition()
                tools.append(tool_def)
            except Exception as e:
                logger.error(f"Failed to get tool definition for {plugin_name}: {e}")
        
        return tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            List of TextContent responses
            
        Raises:
            PluginError: If tool not found or execution fails
        """
        plugin = self.get_plugin(tool_name)
        if not plugin:
            raise PluginError(f"Tool '{tool_name}' not found in registry")
        
        try:
            # Check dependencies before execution
            if not plugin.check_dependencies():
                raise PluginError(f"Tool '{tool_name}' dependencies not satisfied")
            
            result = await plugin.execute(arguments)
            logger.debug(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise PluginError(f"Tool '{tool_name}' execution failed: {e}") from e
    
    def validate_all_plugins(self) -> Dict[str, bool]:
        """Validate all registered plugins.
        
        Returns:
            Dictionary mapping plugin names to validation results
        """
        results = {}
        
        # Validate loaded instances
        for name, plugin in self._plugins.items():
            results[name] = plugin.validate()
        
        # Validate registered classes (create temp instances)
        for name, plugin_class in self._plugin_classes.items():
            if name not in results:  # Not already loaded
                try:
                    temp_instance = plugin_class(self.container)
                    results[name] = temp_instance.validate()
                except Exception as e:
                    logger.error(f"Failed to validate plugin class {name}: {e}")
                    results[name] = False
        
        return results
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin metadata or None if not found
        """
        return self._metadata.get(plugin_name)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the plugin registry.
        
        Returns:
            Dictionary with registry statistics and information
        """
        return {
            "total_registered_classes": len(self._plugin_classes),
            "total_loaded_instances": len(self._plugins),
            "loaded_plugins": list(self._loaded_plugins),
            "available_plugins": self.list_plugins(),
            "tags": {tag: list(plugins) for tag, plugins in self._tags.items()},
            "validation_status": self.validate_all_plugins()
        }
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all registered plugin classes.
        
        Returns:
            Dictionary mapping plugin names to load success status
        """
        results = {}
        
        for plugin_name in self._plugin_classes:
            if plugin_name not in self._loaded_plugins:
                plugin = self.load_plugin(plugin_name)
                results[plugin_name] = plugin is not None
            else:
                results[plugin_name] = True  # Already loaded
        
        logger.info(f"Loaded {sum(results.values())}/{len(results)} plugins")
        return results
    
    def unload_all_plugins(self) -> None:
        """Unload all loaded plugins."""
        plugin_names = list(self._loaded_plugins)
        for name in plugin_names:
            self.unregister_plugin(name)
        
        logger.info("All plugins unloaded")
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            # Unload if currently loaded
            if plugin_name in self._loaded_plugins:
                self.unregister_plugin(plugin_name)
            
            # Load again
            plugin = self.load_plugin(plugin_name)
            return plugin is not None
            
        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry(container=None) -> PluginRegistry:
    """Get the global plugin registry instance.
    
    Args:
        container: Optional service container for dependency injection
        
    Returns:
        Global plugin registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry(container)
    return _global_registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry (mainly for testing)."""
    global _global_registry
    if _global_registry:
        _global_registry.unload_all_plugins()
    _global_registry = None