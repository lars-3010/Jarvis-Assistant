"""
Extension interfaces for Jarvis Assistant.

This module defines the abstract base classes for all extensions in the system,
enabling modular functionality that can be loaded dynamically.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel
import mcp.types as types
from jarvis.core.container import ServiceContainer


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
    error_details: Optional[str] = None
    last_check: Optional[float] = None
    dependencies_healthy: bool = True
    resource_usage: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True


class ExtensionMetadata(BaseModel):
    """Extension metadata model."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = []
    required_services: List[str] = []
    optional_services: List[str] = []
    configuration_schema: Dict[str, Any] = {}
    
    class Config:
        extra = "allow"


class MCPTool(BaseModel):
    """MCP tool definition for extensions."""
    name: str
    description: str
    input_schema: Dict[str, Any]
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
    def get_tools(self) -> List[MCPTool]:
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
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
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
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this extension.
        
        Returns:
            JSON schema for extension configuration
        """
        return {}
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
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
    def list_available_extensions(self) -> List[str]:
        """List all available extensions.
        
        Returns:
            List of extension names
        """
        pass
    
    @abstractmethod
    def list_loaded_extensions(self) -> List[str]:
        """List currently loaded extensions.
        
        Returns:
            List of loaded extension names
        """
        pass
    
    @abstractmethod
    def get_extension(self, extension_name: str) -> Optional[IExtension]:
        """Get a loaded extension by name.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension instance or None if not loaded
        """
        pass
    
    @abstractmethod
    def get_all_tools(self) -> List[MCPTool]:
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