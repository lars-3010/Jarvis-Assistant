"""
Extension registry for Jarvis Assistant.

This module provides a centralized registry for managing extension state,
relationships, and metadata.
"""

import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging

from jarvis.extensions.interfaces import (
    IExtension, ExtensionMetadata, ExtensionHealth, ExtensionStatus, MCPTool
)
from jarvis.extensions.errors import ExtensionError, ExtensionNotFoundError
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


@dataclass
class ExtensionInfo:
    """Information about a registered extension."""
    extension: IExtension
    metadata: ExtensionMetadata
    loaded_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    health_status: Optional[ExtensionHealth] = None
    error_count: int = 0
    last_error: Optional[str] = None


class ExtensionRegistry:
    """Registry for managing extension lifecycle and state."""
    
    def __init__(self):
        """Initialize the extension registry."""
        self._extensions: Dict[str, ExtensionInfo] = {}
        self._tools_cache: Dict[str, str] = {}  # tool_name -> extension_name
        self._dependency_graph: Dict[str, Set[str]] = {}  # extension -> dependencies
        self._reverse_deps: Dict[str, Set[str]] = {}  # extension -> dependents
        self._registry_lock = asyncio.Lock()
        
        logger.info("Extension registry initialized")
    
    async def register_extension(self, extension: IExtension) -> None:
        """Register an extension with the registry.
        
        Args:
            extension: Extension instance to register
            
        Raises:
            ExtensionError: If registration fails
        """
        async with self._registry_lock:
            metadata = extension.get_metadata()
            extension_name = metadata.name
            
            if extension_name in self._extensions:
                logger.warning(f"Extension {extension_name} is already registered, updating")
            
            # Create extension info
            info = ExtensionInfo(
                extension=extension,
                metadata=metadata
            )
            
            # Register the extension
            self._extensions[extension_name] = info
            
            # Update dependency graph
            self._update_dependency_graph(extension_name, metadata.dependencies)
            
            # Cache tools
            self._cache_extension_tools(extension_name, extension.get_tools())
            
            # Initial health check
            await self._update_health_status(extension_name)
            
            logger.info(f"Registered extension: {extension_name} v{metadata.version}")
    
    async def unregister_extension(self, extension_name: str) -> None:
        """Unregister an extension from the registry.
        
        Args:
            extension_name: Name of the extension to unregister
        """
        async with self._registry_lock:
            if extension_name not in self._extensions:
                logger.warning(f"Extension {extension_name} is not registered")
                return
            
            # Check for dependents
            dependents = self._reverse_deps.get(extension_name, set())
            if dependents:
                logger.warning(f"Extension {extension_name} has dependents: {dependents}")
            
            # Remove from registry
            del self._extensions[extension_name]
            
            # Clean up dependency graph
            self._remove_from_dependency_graph(extension_name)
            
            # Clean up tools cache
            self._uncache_extension_tools(extension_name)
            
            logger.info(f"Unregistered extension: {extension_name}")
    
    def get_extension_info(self, extension_name: str) -> Optional[ExtensionInfo]:
        """Get information about a registered extension.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            ExtensionInfo or None if not found
        """
        return self._extensions.get(extension_name)
    
    def get_extension(self, extension_name: str) -> Optional[IExtension]:
        """Get a registered extension instance.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            Extension instance or None if not found
        """
        info = self._extensions.get(extension_name)
        return info.extension if info else None
    
    def list_extensions(self) -> List[str]:
        """List all registered extension names.
        
        Returns:
            List of extension names
        """
        return list(self._extensions.keys())
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from registered extensions.
        
        Returns:
            List of all MCP tools
        """
        tools = []
        for info in self._extensions.values():
            try:
                tools.extend(info.extension.get_tools())
            except Exception as e:
                logger.error(f"Error getting tools from {info.metadata.name}: {e}")
                info.error_count += 1
                info.last_error = str(e)
        
        return tools
    
    def find_tool_extension(self, tool_name: str) -> Optional[str]:
        """Find which extension provides a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Extension name or None if not found
        """
        return self._tools_cache.get(tool_name)
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Route a tool call to the appropriate extension.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ExtensionNotFoundError: If tool or extension not found
            ExtensionError: If tool execution fails
        """
        extension_name = self._tools_cache.get(tool_name)
        if not extension_name:
            raise ExtensionNotFoundError(f"Tool {tool_name} not found", tool_name)
        
        info = self._extensions.get(extension_name)
        if not info:
            raise ExtensionNotFoundError(f"Extension {extension_name} not found", extension_name)
        
        try:
            result = await info.extension.handle_tool_call(tool_name, arguments)
            return result
        except Exception as e:
            info.error_count += 1
            info.last_error = str(e)
            logger.error(f"Tool {tool_name} execution failed in {extension_name}: {e}")
            raise ExtensionError(f"Tool execution failed: {str(e)}", extension_name, e)
    
    async def check_all_health(self) -> Dict[str, ExtensionHealth]:
        """Check health status of all registered extensions.
        
        Returns:
            Dict mapping extension names to health status
        """
        health_results = {}
        
        for extension_name in self._extensions.keys():
            try:
                await self._update_health_status(extension_name)
                info = self._extensions[extension_name]
                health_results[extension_name] = info.health_status
            except Exception as e:
                logger.error(f"Health check failed for {extension_name}: {e}")
                health_results[extension_name] = ExtensionHealth(
                    status=ExtensionStatus.ERROR,
                    message=f"Health check failed: {str(e)}",
                    error_details=str(e),
                    last_check=time.time()
                )
        
        return health_results
    
    def get_dependency_order(self) -> List[str]:
        """Get extensions in dependency order (dependencies first).
        
        Returns:
            List of extension names in dependency order
        """
        return self._topological_sort()
    
    def get_shutdown_order(self) -> List[str]:
        """Get extensions in shutdown order (reverse dependency order).
        
        Returns:
            List of extension names in shutdown order
        """
        return list(reversed(self._topological_sort()))
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total_extensions = len(self._extensions)
        total_tools = len(self._tools_cache)
        
        # Count by status
        status_counts = {}
        error_extensions = []
        
        for name, info in self._extensions.items():
            if info.health_status:
                status = info.health_status.status
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if status == ExtensionStatus.ERROR:
                    error_extensions.append(name)
        
        return {
            "total_extensions": total_extensions,
            "total_tools": total_tools,
            "status_counts": status_counts,
            "error_extensions": error_extensions,
            "dependency_graph_size": len(self._dependency_graph),
            "registry_lock_locked": self._registry_lock.locked()
        }
    
    def _update_dependency_graph(self, extension_name: str, dependencies: List[str]) -> None:
        """Update the dependency graph for an extension.
        
        Args:
            extension_name: Name of the extension
            dependencies: List of dependency names
        """
        # Update forward dependencies
        self._dependency_graph[extension_name] = set(dependencies)
        
        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self._reverse_deps:
                self._reverse_deps[dep] = set()
            self._reverse_deps[dep].add(extension_name)
    
    def _remove_from_dependency_graph(self, extension_name: str) -> None:
        """Remove an extension from the dependency graph.
        
        Args:
            extension_name: Name of the extension to remove
        """
        # Remove forward dependencies
        dependencies = self._dependency_graph.pop(extension_name, set())
        
        # Remove from reverse dependencies
        for dep in dependencies:
            if dep in self._reverse_deps:
                self._reverse_deps[dep].discard(extension_name)
                if not self._reverse_deps[dep]:
                    del self._reverse_deps[dep]
        
        # Remove as a reverse dependency
        self._reverse_deps.pop(extension_name, None)
    
    def _cache_extension_tools(self, extension_name: str, tools: List[MCPTool]) -> None:
        """Cache tools provided by an extension.
        
        Args:
            extension_name: Name of the extension
            tools: List of tools provided by the extension
        """
        for tool in tools:
            if tool.name in self._tools_cache:
                logger.warning(f"Tool {tool.name} is already provided by {self._tools_cache[tool.name]}")
            
            self._tools_cache[tool.name] = extension_name
    
    def _uncache_extension_tools(self, extension_name: str) -> None:
        """Remove tools from cache for an extension.
        
        Args:
            extension_name: Name of the extension
        """
        tools_to_remove = [
            tool_name for tool_name, ext_name in self._tools_cache.items()
            if ext_name == extension_name
        ]
        
        for tool_name in tools_to_remove:
            del self._tools_cache[tool_name]
    
    async def _update_health_status(self, extension_name: str) -> None:
        """Update health status for an extension.
        
        Args:
            extension_name: Name of the extension
        """
        info = self._extensions.get(extension_name)
        if not info:
            return
        
        try:
            health = info.extension.get_health_status()
            info.health_status = health
            info.last_health_check = datetime.now()
        except Exception as e:
            logger.error(f"Failed to get health status for {extension_name}: {e}")
            info.health_status = ExtensionHealth(
                status=ExtensionStatus.ERROR,
                message=f"Health check failed: {str(e)}",
                error_details=str(e),
                last_check=time.time()
            )
            info.error_count += 1
            info.last_error = str(e)
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort of extensions based on dependencies.
        
        Returns:
            List of extension names in dependency order
        """
        # Kahn's algorithm for topological sorting
        in_degree = {ext: 0 for ext in self._extensions.keys()}
        
        # Calculate in-degrees
        for extension, deps in self._dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find nodes with no incoming edges
        queue = [ext for ext, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove this node and update in-degrees
            for dependent in self._reverse_deps.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(self._extensions):
            logger.warning("Circular dependencies detected in extension graph")
            # Return remaining extensions in arbitrary order
            remaining = set(self._extensions.keys()) - set(result)
            result.extend(remaining)
        
        return result