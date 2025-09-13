"""
Extension loader for Jarvis Assistant.

This module provides functionality to discover, load, and manage extensions
dynamically at runtime.
"""

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from jarvis.core.container import ServiceContainer
from jarvis.extensions.errors import (
    ExtensionAlreadyLoadedError,
    ExtensionDependencyError,
    ExtensionError,
    ExtensionLoadError,
    ExtensionNotFoundError,
)
from jarvis.extensions.interfaces import (
    ExtensionHealth,
    ExtensionMetadata,
    ExtensionStatus,
    IExtension,
    IExtensionManager,
)
from jarvis.utils.config import JarvisSettings
import logging

logger = logging.getLogger(__name__)


class ExtensionLoader(IExtensionManager):
    """Manages loading and unloading of extensions."""

    def __init__(self, settings: JarvisSettings, container: ServiceContainer):
        """Initialize the extension loader.
        
        Args:
            settings: Application settings
            container: Service container for dependency injection
        """
        self.settings = settings
        self.container = container
        self.loaded_extensions: dict[str, IExtension] = {}
        self.extension_metadata: dict[str, ExtensionMetadata] = {}
        self._loading_lock = asyncio.Lock()

        logger.info(f"Extension loader initialized - enabled: {settings.extensions_enabled}")

    async def discover_extensions(self) -> list[str]:
        """Discover available extensions in the extensions directory.
        
        Returns:
            List of extension names that can be loaded
        """
        if not self.settings.extensions_enabled:
            logger.debug("Extensions disabled, returning empty list")
            return []

        extensions_dir = self.settings.get_extensions_directory()
        if not extensions_dir.exists():
            logger.warning(f"Extensions directory does not exist: {extensions_dir}")
            return []

        discovered = []
        logger.debug(f"Scanning for extensions in: {extensions_dir}")

        # Look for directories containing extension modules
        for item in extensions_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                # Check for main.py or __init__.py
                main_file = item / "main.py"
                init_file = item / "__init__.py"

                if main_file.exists() or init_file.exists():
                    discovered.append(item.name)
                    logger.debug(f"Discovered extension: {item.name}")

        logger.info(f"Discovered {len(discovered)} extensions: {discovered}")
        return discovered

    async def load_extension(self, extension_name: str) -> IExtension:
        """Load an extension by name.
        
        Args:
            extension_name: Name of the extension to load
            
        Returns:
            Loaded extension instance
            
        Raises:
            ExtensionError: If loading fails
        """
        if not self.settings.extensions_enabled:
            raise ExtensionError("Extensions are disabled", extension_name)

        async with self._loading_lock:
            if extension_name in self.loaded_extensions:
                raise ExtensionAlreadyLoadedError(f"Extension {extension_name} is already loaded", extension_name)

            logger.info(f"Loading extension: {extension_name}")

            try:
                # Discover the extension
                extension_path = self._find_extension_path(extension_name)
                if not extension_path:
                    raise ExtensionNotFoundError(f"Extension {extension_name} not found", extension_name)

                # Load the extension module
                extension_module = self._load_extension_module(extension_name, extension_path)

                # Get the extension class
                extension_class = self._get_extension_class(extension_module, extension_name)

                # Create extension instance
                extension_instance = extension_class()

                # Validate dependencies
                metadata = extension_instance.get_metadata()
                self._validate_dependencies(metadata)

                # Initialize the extension
                await extension_instance.initialize(self.container)

                # Store the loaded extension
                self.loaded_extensions[extension_name] = extension_instance
                self.extension_metadata[extension_name] = metadata

                logger.info(f"Successfully loaded extension: {extension_name} v{metadata.version}")
                return extension_instance

            except Exception as e:
                logger.error(f"Failed to load extension {extension_name}: {e}")
                if isinstance(e, ExtensionError):
                    raise
                raise ExtensionLoadError(f"Failed to load extension {extension_name}: {e!s}", extension_name, e)

    async def unload_extension(self, extension_name: str) -> None:
        """Unload an extension.
        
        Args:
            extension_name: Name of the extension to unload
        """
        async with self._loading_lock:
            if extension_name not in self.loaded_extensions:
                logger.warning(f"Extension {extension_name} is not loaded")
                return

            logger.info(f"Unloading extension: {extension_name}")

            try:
                extension = self.loaded_extensions[extension_name]
                await extension.shutdown()

                # Remove from loaded extensions
                del self.loaded_extensions[extension_name]
                if extension_name in self.extension_metadata:
                    del self.extension_metadata[extension_name]

                logger.info(f"Successfully unloaded extension: {extension_name}")

            except Exception as e:
                logger.error(f"Error unloading extension {extension_name}: {e}")
                # Still remove it from the registry even if shutdown failed
                self.loaded_extensions.pop(extension_name, None)
                self.extension_metadata.pop(extension_name, None)

    async def reload_extension(self, extension_name: str) -> IExtension:
        """Reload an extension.
        
        Args:
            extension_name: Name of the extension to reload
            
        Returns:
            Reloaded extension instance
        """
        logger.info(f"Reloading extension: {extension_name}")

        # Unload if currently loaded
        if extension_name in self.loaded_extensions:
            await self.unload_extension(extension_name)

        # Reload the module from disk
        self._reload_extension_module(extension_name)

        # Load the extension again
        return await self.load_extension(extension_name)

    def list_available_extensions(self) -> list[str]:
        """List all available extensions.
        
        Returns:
            List of extension names
        """
        return asyncio.run(self.discover_extensions())

    def list_loaded_extensions(self) -> list[str]:
        """List currently loaded extensions.
        
        Returns:
            List of loaded extension names
        """
        return list(self.loaded_extensions.keys())

    def get_extension(self, extension_name: str) -> IExtension | None:
        """Get a loaded extension by name.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension instance or None if not loaded
        """
        return self.loaded_extensions.get(extension_name)

    def get_all_tools(self) -> list[Any]:
        """Get all tools from loaded extensions.
        
        Returns:
            List of all MCP tools from loaded extensions
        """
        tools = []
        for extension in self.loaded_extensions.values():
            tools.extend(extension.get_tools())
        return tools

    def get_extension_health(self) -> dict[str, ExtensionHealth]:
        """Get health status of all loaded extensions.
        
        Returns:
            Dict mapping extension names to their health status
        """
        health = {}
        for name, extension in self.loaded_extensions.items():
            try:
                health[name] = extension.get_health_status()
            except Exception as e:
                health[name] = ExtensionHealth(
                    status=ExtensionStatus.ERROR,
                    message=f"Error getting health status: {e!s}",
                    error_details=str(e)
                )
        return health

    async def load_auto_extensions(self) -> None:
        """Load extensions specified in auto_load configuration."""
        if not self.settings.extensions_enabled:
            logger.debug("Extensions disabled, skipping auto-load")
            return

        auto_load = self.settings.extensions_auto_load
        if not auto_load:
            logger.debug("No extensions configured for auto-load")
            return

        logger.info(f"Auto-loading extensions: {auto_load}")

        for extension_name in auto_load:
            try:
                await self.load_extension(extension_name)
                logger.info(f"Auto-loaded extension: {extension_name}")
            except Exception as e:
                logger.error(f"Failed to auto-load extension {extension_name}: {e}")

    def _find_extension_path(self, extension_name: str) -> Path | None:
        """Find the path to an extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Path to extension directory or None if not found
        """
        extensions_dir = self.settings.get_extensions_directory()
        extension_path = extensions_dir / extension_name

        if extension_path.exists() and extension_path.is_dir():
            return extension_path

        return None

    def _load_extension_module(self, extension_name: str, extension_path: Path):
        """Load an extension module from disk.
        
        Args:
            extension_name: Name of the extension
            extension_path: Path to the extension directory
            
        Returns:
            Loaded module
        """
        # Try main.py first, then __init__.py
        main_file = extension_path / "main.py"
        init_file = extension_path / "__init__.py"

        module_file = main_file if main_file.exists() else init_file
        if not module_file.exists():
            raise ExtensionLoadError(f"No main.py or __init__.py found in {extension_path}", extension_name)

        # Create module spec
        module_name = f"jarvis.extensions.{extension_name}"
        spec = importlib.util.spec_from_file_location(module_name, module_file)

        if spec is None or spec.loader is None:
            raise ExtensionLoadError(f"Could not create module spec for {extension_name}", extension_name)

        # Load the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module

    def _get_extension_class(self, module, extension_name: str) -> type[IExtension]:
        """Get the extension class from a module.
        
        Args:
            module: Loaded module
            extension_name: Name of the extension
            
        Returns:
            Extension class
        """
        # Look for Extension class, or {ExtensionName}Extension class
        possible_names = [
            "Extension",
            f"{extension_name.title()}Extension",
            f"{extension_name}Extension"
        ]

        for class_name in possible_names:
            if hasattr(module, class_name):
                extension_class = getattr(module, class_name)
                if issubclass(extension_class, IExtension):
                    return extension_class

        raise ExtensionLoadError(f"No valid extension class found in {extension_name}", extension_name)

    def _validate_dependencies(self, metadata: ExtensionMetadata) -> None:
        """Validate extension dependencies.
        
        Args:
            metadata: Extension metadata
            
        Raises:
            ExtensionDependencyError: If dependencies are not met
        """
        # Check required services
        for service_name in metadata.required_services:
            # This is a simplified check - in a full implementation,
            # you would check if the service is available in the container
            logger.debug(f"Checking required service: {service_name}")

        # Check extension dependencies
        for dep_name in metadata.dependencies:
            if dep_name not in self.loaded_extensions:
                raise ExtensionDependencyError(
                    f"Required extension dependency {dep_name} is not loaded",
                    metadata.name
                )

    def _reload_extension_module(self, extension_name: str) -> None:
        """Reload an extension module from disk.
        
        Args:
            extension_name: Name of the extension
        """
        module_name = f"jarvis.extensions.{extension_name}"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

    async def shutdown_all(self) -> None:
        """Shutdown all loaded extensions."""
        logger.info("Shutting down all extensions")

        for extension_name in list(self.loaded_extensions.keys()):
            await self.unload_extension(extension_name)

        logger.info("All extensions shut down")
