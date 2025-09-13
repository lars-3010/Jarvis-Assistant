"""
Extension lifecycle manager for Jarvis Assistant.

This module provides high-level management of extension lifecycle,
coordinating loading, health monitoring, and shutdown.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from jarvis.core.container import ServiceContainer
from jarvis.extensions.errors import ExtensionError
from jarvis.extensions.interfaces import ExtensionStatus, IExtension, MCPTool
from jarvis.extensions.loader import ExtensionLoader
from jarvis.extensions.registry import ExtensionRegistry
from jarvis.utils.config import JarvisSettings
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class ExtensionManager:
    """High-level extension lifecycle manager."""

    def __init__(self, settings: JarvisSettings, container: ServiceContainer):
        """Initialize the extension manager.
        
        Args:
            settings: Application settings
            container: Service container for dependency injection
        """
        self.settings = settings
        self.container = container
        self.loader = ExtensionLoader(settings, container)
        self.registry = ExtensionRegistry()

        # Health monitoring
        self._health_check_task: asyncio.Task | None = None
        self._health_check_interval = 30  # seconds
        self._shutdown_event = asyncio.Event()

        # State tracking
        self._initialized = False
        self._shutting_down = False

        logger.info("Extension manager initialized")

    async def initialize(self) -> None:
        """Initialize the extension system."""
        if self._initialized:
            logger.warning("Extension manager already initialized")
            return

        if not self.settings.extensions_enabled:
            logger.info("Extensions disabled, skipping initialization")
            return

        logger.info("Initializing extension system")

        try:
            # Load auto-load extensions
            await self.loader.load_auto_extensions()

            # Register loaded extensions with registry
            for name, extension in self.loader.loaded_extensions.items():
                await self.registry.register_extension(extension)

            # Start health monitoring
            self._start_health_monitoring()

            self._initialized = True
            logger.info(f"Extension system initialized with {len(self.loader.loaded_extensions)} extensions")

        except Exception as e:
            logger.error(f"Failed to initialize extension system: {e}")
            raise ExtensionError("Extension system initialization failed", cause=e)

    async def shutdown(self) -> None:
        """Shutdown the extension system."""
        if self._shutting_down:
            logger.warning("Extension manager already shutting down")
            return

        self._shutting_down = True
        logger.info("Shutting down extension system")

        try:
            # Stop health monitoring
            self._shutdown_event.set()
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Shutdown extensions in dependency order
            shutdown_order = self.registry.get_shutdown_order()
            logger.info(f"Shutting down extensions in order: {shutdown_order}")

            for extension_name in shutdown_order:
                try:
                    await self.loader.unload_extension(extension_name)
                    await self.registry.unregister_extension(extension_name)
                except Exception as e:
                    logger.error(f"Error shutting down extension {extension_name}: {e}")

            self._initialized = False
            logger.info("Extension system shutdown complete")

        except Exception as e:
            logger.error(f"Error during extension system shutdown: {e}")
            raise

    async def load_extension(self, extension_name: str) -> IExtension:
        """Load and register an extension.
        
        Args:
            extension_name: Name of the extension to load
            
        Returns:
            Loaded extension instance
        """
        if not self.settings.extensions_enabled:
            raise ExtensionError("Extensions are disabled")

        logger.info(f"Loading extension: {extension_name}")

        # Load the extension
        extension = await self.loader.load_extension(extension_name)

        # Register with registry
        await self.registry.register_extension(extension)

        logger.info(f"Successfully loaded and registered extension: {extension_name}")
        return extension

    async def unload_extension(self, extension_name: str) -> None:
        """Unload and unregister an extension.
        
        Args:
            extension_name: Name of the extension to unload
        """
        logger.info(f"Unloading extension: {extension_name}")

        # Unregister from registry
        await self.registry.unregister_extension(extension_name)

        # Unload the extension
        await self.loader.unload_extension(extension_name)

        logger.info(f"Successfully unloaded extension: {extension_name}")

    async def reload_extension(self, extension_name: str) -> IExtension:
        """Reload an extension.
        
        Args:
            extension_name: Name of the extension to reload
            
        Returns:
            Reloaded extension instance
        """
        logger.info(f"Reloading extension: {extension_name}")

        # Unload if loaded
        if extension_name in self.registry.list_extensions():
            await self.unload_extension(extension_name)

        # Load again
        return await self.load_extension(extension_name)

    def get_extension(self, extension_name: str) -> IExtension | None:
        """Get a loaded extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension instance or None if not found
        """
        return self.registry.get_extension(extension_name)

    def list_extensions(self) -> dict[str, Any]:
        """List all extensions with their status.
        
        Returns:
            Dict with extension information
        """
        available = self.loader.list_available_extensions()
        loaded = self.registry.list_extensions()

        extensions_info = {}

        for name in available:
            info = {
                "available": True,
                "loaded": name in loaded,
                "metadata": None,
                "health": None
            }

            if name in loaded:
                ext_info = self.registry.get_extension_info(name)
                if ext_info:
                    info["metadata"] = ext_info.metadata.model_dump()
                    info["health"] = ext_info.health_status.model_dump() if ext_info.health_status else None

            extensions_info[name] = info

        return extensions_info

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from loaded extensions.
        
        Returns:
            List of all MCP tools
        """
        return self.registry.get_all_tools()

    async def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Handle a tool call from extensions.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        return await self.registry.handle_tool_call(tool_name, arguments)

    async def check_health(self) -> dict[str, Any]:
        """Check health of the extension system.
        
        Returns:
            Health status information
        """
        if not self.settings.extensions_enabled:
            return {
                "enabled": False,
                "message": "Extensions disabled"
            }

        extension_health = await self.registry.check_all_health()
        registry_stats = self.registry.get_registry_stats()

        # Count healthy extensions
        healthy_count = sum(
            1 for health in extension_health.values()
            if health.status == ExtensionStatus.ACTIVE
        )

        total_count = len(extension_health)

        return {
            "enabled": True,
            "initialized": self._initialized,
            "total_extensions": total_count,
            "healthy_extensions": healthy_count,
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 100,
            "extensions": extension_health,
            "registry_stats": registry_stats,
            "loader_stats": {
                "loaded_count": len(self.loader.loaded_extensions),
                "auto_load_configured": len(self.settings.extensions_auto_load)
            }
        }

    def get_extension_config(self, extension_name: str) -> dict[str, Any]:
        """Get configuration for a specific extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension configuration
        """
        return self.settings.extensions_config.get(extension_name, {})

    def _start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            logger.debug("Started extension health monitoring")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        logger.debug("Extension health monitoring started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for interval or shutdown event
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._health_check_interval
                )
                break  # Shutdown requested
            except TimeoutError:
                # Time to do health check
                pass

            try:
                # Perform health checks
                await self.registry.check_all_health()
                logger.debug("Extension health check completed")
            except Exception as e:
                logger.error(f"Error during extension health check: {e}")

        logger.debug("Extension health monitoring stopped")

    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for extension lifecycle.
        
        Usage:
            async with extension_manager.managed_lifecycle():
                # Extensions are initialized and ready
                pass
            # Extensions are shut down
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    def is_extension_loaded(self, extension_name: str) -> bool:
        """Check if an extension is loaded.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            True if extension is loaded
        """
        return extension_name in self.registry.list_extensions()

    def get_tool_provider(self, tool_name: str) -> str | None:
        """Get the extension that provides a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Extension name or None if not found
        """
        return self.registry.find_tool_extension(tool_name)

    async def validate_extension_dependencies(self) -> dict[str, list[str]]:
        """Validate all extension dependencies.
        
        Returns:
            Dict mapping extension names to missing dependencies
        """
        missing_deps = {}
        loaded_extensions = set(self.registry.list_extensions())

        for extension_name in loaded_extensions:
            info = self.registry.get_extension_info(extension_name)
            if info:
                missing = []
                for dep in info.metadata.dependencies:
                    if dep not in loaded_extensions:
                        missing.append(dep)

                if missing:
                    missing_deps[extension_name] = missing

        return missing_deps
