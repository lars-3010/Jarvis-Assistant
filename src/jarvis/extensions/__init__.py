"""
Jarvis Assistant Extension System.

This package provides a plugin architecture for extending Jarvis functionality
with AI capabilities and other features.

Key Components:
- IExtension: Base interface for all extensions
- ExtensionManager: High-level lifecycle management
- ExtensionLoader: Dynamic loading and unloading
- ExtensionRegistry: State and relationship management
- ExtensionValidator: Configuration and requirement validation

Usage:
    from jarvis.extensions import ExtensionManager
    from jarvis.core.container import ServiceContainer
    from jarvis.utils.config import get_settings
    
    settings = get_settings()
    container = ServiceContainer(settings)
    
    # Initialize extension system
    manager = ExtensionManager(settings, container)
    
    async with manager.managed_lifecycle():
        # Extensions are loaded and ready
        tools = manager.get_all_tools()
        result = await manager.handle_tool_call("tool-name", {"arg": "value"})
    # Extensions are shut down automatically
"""

from jarvis.extensions.interfaces import (
    IExtension,
    IExtensionManager,
    ExtensionStatus,
    ExtensionHealth,
    ExtensionMetadata,
    MCPTool,
)

from jarvis.extensions.manager import ExtensionManager
from jarvis.extensions.loader import ExtensionLoader
from jarvis.extensions.registry import ExtensionRegistry
from jarvis.extensions.validation import ExtensionValidator

from jarvis.extensions.errors import (
    ExtensionError,
    ExtensionLoadError,
    ExtensionInitializationError,
    ExtensionConfigurationError,
    ExtensionDependencyError,
    ExtensionToolError,
    ExtensionNotFoundError,
    ExtensionAlreadyLoadedError,
)

__all__ = [
    # Core interfaces
    "IExtension",
    "IExtensionManager", 
    "ExtensionStatus",
    "ExtensionHealth",
    "ExtensionMetadata",
    "MCPTool",
    
    # Main components
    "ExtensionManager",
    "ExtensionLoader",
    "ExtensionRegistry",
    "ExtensionValidator",
    
    # Exceptions
    "ExtensionError",
    "ExtensionLoadError",
    "ExtensionInitializationError", 
    "ExtensionConfigurationError",
    "ExtensionDependencyError",
    "ExtensionToolError",
    "ExtensionNotFoundError",
    "ExtensionAlreadyLoadedError",
]

# Extension system version
__version__ = "0.1.0"