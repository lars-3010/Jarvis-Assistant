"""
Base classes for MCP Tool Plugins.

This module defines the abstract base classes and interfaces for MCP tool plugins,
providing a standardized way to implement discoverable and executable tools.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jarvis.core.interfaces import *
from jarvis.utils.logging import setup_logging
from mcp import types

logger = setup_logging(__name__)


@dataclass
class PluginMetadata:
    """Metadata for MCP tool plugins."""

    name: str
    version: str
    description: str
    author: str | None = None
    license: str | None = None
    dependencies: list[str] | None = None
    tags: list[str] | None = None
    min_jarvis_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "dependencies": self.dependencies or [],
            "tags": self.tags or [],
            "min_jarvis_version": self.min_jarvis_version
        }


class MCPToolPlugin(ABC):
    """Abstract base class for all MCP tool plugins.
    
    This class defines the interface that all MCP tool plugins must implement
    to be discoverable and executable by the plugin system.
    """

    def __init__(self, container=None):
        """Initialize the plugin.
        
        Args:
            container: Optional service container for dependency injection
        """
        self.container = container
        self._metadata: PluginMetadata | None = None
        self._validated = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the unique name of this plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a human-readable description of this plugin."""
        pass

    @property
    def version(self) -> str:
        """Get the version of this plugin."""
        return "1.0.0"

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        if self._metadata is None:
            self._metadata = PluginMetadata(
                name=self.name,
                version=self.version,
                description=self.description,
                author=getattr(self, 'author', None),
                license=getattr(self, 'license', None),
                dependencies=getattr(self, 'dependencies', None),
                tags=getattr(self, 'tags', None),
                min_jarvis_version=getattr(self, 'min_jarvis_version', None)
            )
        return self._metadata

    @abstractmethod
    def get_tool_definition(self) -> types.Tool:
        """Get the MCP tool definition for this plugin.
        
        Returns:
            MCP Tool definition with name, description, and input schema
        """
        pass

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Execute the plugin with the given arguments.
        
        Args:
            arguments: Dictionary of arguments passed to the tool
            
        Returns:
            List of TextContent responses
            
        Raises:
            PluginError: If execution fails
        """
        pass

    def validate(self) -> bool:
        """Validate that the plugin is properly configured.
        
        Returns:
            True if plugin is valid, False otherwise
        """
        try:
            # Check required properties
            if not self.name or not isinstance(self.name, str):
                logger.error(f"Plugin {self.__class__.__name__} has invalid name")
                return False

            if not self.description or not isinstance(self.description, str):
                logger.error(f"Plugin {self.name} has invalid description")
                return False

            # Validate tool definition
            tool_def = self.get_tool_definition()
            if not isinstance(tool_def, types.Tool):
                logger.error(f"Plugin {self.name} has invalid tool definition")
                return False

            # Check execute method signature
            sig = inspect.signature(self.execute)
            if len(sig.parameters) != 1:
                logger.error(f"Plugin {self.name} execute method has wrong signature")
                return False

            self._validated = True
            logger.debug(f"Plugin {self.name} validation passed")
            return True

        except Exception as e:
            logger.error(f"Plugin {self.name} validation failed: {e}")
            return False

    @property
    def is_validated(self) -> bool:
        """Check if plugin has been validated."""
        return self._validated

    def get_required_services(self) -> list[type]:
        """Get list of service interfaces required by this plugin.
        
        Returns:
            List of service interface classes
        """
        return []

    def check_dependencies(self) -> bool:
        """Check if all plugin dependencies are available.
        
        Returns:
            True if all dependencies are satisfied
        """
        if not self.container:
            return True  # No dependency checking without container

        try:
            for service_type in self.get_required_services():
                service = self.container.get(service_type)
                if service is None:
                    logger.warning(f"Plugin {self.name} missing required service: {service_type.__name__}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Plugin {self.name} dependency check failed: {e}")
            return False

    def on_load(self) -> None:
        """Called when the plugin is loaded into the registry."""
        logger.debug(f"Plugin {self.name} loaded")

    def on_unload(self) -> None:
        """Called when the plugin is unloaded from the registry."""
        logger.debug(f"Plugin {self.name} unloaded")

    def __repr__(self) -> str:
        """String representation of the plugin."""
        return f"<{self.__class__.__name__}: {self.name} v{self.version}>"


class SearchPlugin(MCPToolPlugin):
    """Base class for search-related plugins."""

    @property
    def tags(self) -> list[str]:
        """Default tags for search plugins."""
        return ["search", "query"]

    def get_required_services(self) -> list[type]:
        """Search plugins typically need vector searcher."""
        return [IVectorSearcher]


class GraphPlugin(MCPToolPlugin):
    """Base class for graph-related plugins."""

    @property
    def tags(self) -> list[str]:
        """Default tags for graph plugins."""
        return ["graph", "relationships"]

    def get_required_services(self) -> list[type]:
        """Graph plugins typically need graph database."""
        return [IGraphDatabase]


class VaultPlugin(MCPToolPlugin):
    """Base class for vault-related plugins."""

    @property
    def tags(self) -> list[str]:
        """Default tags for vault plugins."""
        return ["vault", "files"]

    def get_required_services(self) -> list[type]:
        """Vault plugins typically need vault reader."""
        return [IVaultReader]


class UtilityPlugin(MCPToolPlugin):
    """Base class for utility plugins."""

    @property
    def tags(self) -> list[str]:
        """Default tags for utility plugins."""
        return ["utility", "helper"]
