"""
Extension interfaces for Jarvis Assistant.

This module defines the abstract base classes for all extensions in the system,
enabling modular functionality that can be loaded dynamically.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from jarvis.core.container import ServiceContainer
from mcp import types


class ExtensionStatus(str, Enum):
    """Extension status enumeration."""
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class ExtensionHealth(BaseModel):
    """Extension health status model."""
    status: ExtensionStatus
    message: str = ""
    error_details: str | None = None
    last_check: float | None = None
    dependencies_healthy: bool = True
    resource_usage: dict[str, Any] = Field(default_factory=dict)

    # Pydantic v2 configuration
    model_config = ConfigDict(use_enum_values=True)


class ExtensionMetadata(BaseModel):
    """Extension metadata model."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: list[str] = Field(default_factory=list)
    required_services: list[str] = Field(default_factory=list)
    optional_services: list[str] = Field(default_factory=list)
    configuration_schema: dict[str, Any] = Field(default_factory=dict)

    # Pydantic v2 configuration
    model_config = ConfigDict(extra="allow")


class MCPTool(BaseModel):
    """MCP tool definition for extensions."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Any  # Callable that handles the tool execution

    def to_mcp_tool(self) -> types.Tool:
        """Convert to MCP types.Tool."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema
        )


class IExtension(ABC):
    """Abstract base class for all Jarvis extensions."""

    @abstractmethod
    def get_metadata(self) -> ExtensionMetadata:
        """Return extension metadata.
        
        Returns:
            ExtensionMetadata with extension information
        """
        pass

    @abstractmethod
    async def initialize(self, container: ServiceContainer) -> None:
        """Initialize the extension with access to core services.
        
        Args:
            container: Service container for dependency injection
            
        Raises:
            ExtensionError: If initialization fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of extension resources.
        
        This method should cleanup any resources, close connections,
        and prepare the extension for removal.
        """
        pass

    @abstractmethod
    def get_tools(self) -> list[MCPTool]:
        """Return MCP tools provided by this extension.
        
        Returns:
            List of MCPTool instances
        """
        pass

    @abstractmethod
    def get_health_status(self) -> ExtensionHealth:
        """Return current health status of the extension.
        
        Returns:
            ExtensionHealth with current status
        """
        pass

    @abstractmethod
    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        """Handle a tool call from the MCP server.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            
        Returns:
            List of MCP content responses
            
        Raises:
            ExtensionError: If tool execution fails
        """
        pass

    def get_configuration_schema(self) -> dict[str, Any]:
        """Return configuration schema for this extension.
        
        Returns:
            JSON schema for extension configuration
        """
        return {}

    def validate_configuration(self, config: dict[str, Any]) -> bool:
        """Validate extension configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        return True


class IExtensionManager(ABC):
    """Abstract interface for extension management."""

    @abstractmethod
    async def load_extension(self, extension_name: str) -> IExtension:
        """Load an extension by name.
        
        Args:
            extension_name: Name of the extension to load
            
        Returns:
            Loaded extension instance
            
        Raises:
            ExtensionError: If loading fails
        """
        pass

    @abstractmethod
    async def unload_extension(self, extension_name: str) -> None:
        """Unload an extension.
        
        Args:
            extension_name: Name of the extension to unload
        """
        pass

    @abstractmethod
    def list_available_extensions(self) -> list[str]:
        """List all available extensions.
        
        Returns:
            List of extension names
        """
        pass

    @abstractmethod
    def list_loaded_extensions(self) -> list[str]:
        """List currently loaded extensions.
        
        Returns:
            List of loaded extension names
        """
        pass

    @abstractmethod
    def get_extension(self, extension_name: str) -> IExtension | None:
        """Get a loaded extension by name.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension instance or None if not loaded
        """
        pass

    @abstractmethod
    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from loaded extensions.
        
        Returns:
            List of all MCP tools
        """
        pass

    @abstractmethod
    async def reload_extension(self, extension_name: str) -> IExtension:
        """Reload an extension.
        
        Args:
            extension_name: Name of the extension to reload
            
        Returns:
            Reloaded extension instance
        """
        pass
