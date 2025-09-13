"""
Plugin Discovery System for MCP Tool Plugins.

This module provides automatic discovery and loading of MCP tool plugins
from various sources including directories, Python modules, and packages.
"""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any

from jarvis.mcp.plugins.base import MCPToolPlugin
from jarvis.mcp.plugins.registry import PluginRegistry
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class PluginDiscovery:
    """Automatic discovery and loading of MCP tool plugins."""

    def __init__(self, registry: PluginRegistry):
        """Initialize plugin discovery.
        
        Args:
            registry: Plugin registry to register discovered plugins
        """
        self.registry = registry
        self._discovered_modules: set[str] = set()
        self._discovery_paths: list[Path] = []

    def add_discovery_path(self, path: Path) -> None:
        """Add a path to search for plugins.
        
        Args:
            path: Directory path to search for plugins
        """
        if path.exists() and path.is_dir():
            self._discovery_paths.append(path)
            logger.info(f"Added plugin discovery path: {path}")
        else:
            logger.warning(f"Plugin discovery path does not exist: {path}")

    def discover_from_directory(self, directory: Path, recursive: bool = True) -> list[type[MCPToolPlugin]]:
        """Discover plugins from a directory.
        
        Args:
            directory: Directory to search for plugins
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of discovered plugin classes
        """
        discovered_plugins = []

        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return discovered_plugins

        logger.info(f"Discovering plugins in directory: {directory}")

        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(directory.glob(pattern))

        for py_file in python_files:
            if py_file.name.startswith('_'):
                continue  # Skip private files

            try:
                plugins = self._load_plugins_from_file(py_file)
                discovered_plugins.extend(plugins)
            except Exception as e:
                logger.error(f"Failed to load plugins from {py_file}: {e}")

        logger.info(f"Discovered {len(discovered_plugins)} plugins in {directory}")
        return discovered_plugins

    def discover_from_package(self, package_name: str) -> list[type[MCPToolPlugin]]:
        """Discover plugins from a Python package.
        
        Args:
            package_name: Name of the package to search
            
        Returns:
            List of discovered plugin classes
        """
        discovered_plugins = []

        try:
            package = importlib.import_module(package_name)

            # If package has __path__, it's a package directory
            if hasattr(package, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
                    full_module_name = f"{package_name}.{modname}"
                    try:
                        module = importlib.import_module(full_module_name)
                        plugins = self._extract_plugins_from_module(module)
                        discovered_plugins.extend(plugins)
                    except Exception as e:
                        logger.error(f"Failed to load module {full_module_name}: {e}")
            else:
                # Single module
                plugins = self._extract_plugins_from_module(package)
                discovered_plugins.extend(plugins)

            logger.info(f"Discovered {len(discovered_plugins)} plugins in package {package_name}")

        except ImportError as e:
            logger.error(f"Failed to import package {package_name}: {e}")

        return discovered_plugins

    def discover_builtin_plugins(self) -> list[type[MCPToolPlugin]]:
        """Discover built-in plugins from the tools package.
        
        Returns:
            List of discovered built-in plugin classes
        """
        try:
            # Import the tools package from our plugin system
            from jarvis.mcp.plugins.tools import get_builtin_plugins
            return get_builtin_plugins()
        except ImportError:
            logger.warning("Built-in plugins package not found")
            return []

    def _load_plugins_from_file(self, file_path: Path) -> list[type[MCPToolPlugin]]:
        """Load plugins from a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of plugin classes found in the file
        """
        module_name = file_path.stem

        # Skip if already loaded
        if str(file_path) in self._discovered_modules:
            return []

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not create module spec for {file_path}")
                return []

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules to handle imports within the module
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Extract plugins
            plugins = self._extract_plugins_from_module(module)
            self._discovered_modules.add(str(file_path))

            return plugins

        except Exception as e:
            logger.error(f"Failed to load module from {file_path}: {e}")
            return []

    def _extract_plugins_from_module(self, module) -> list[type[MCPToolPlugin]]:
        """Extract plugin classes from a loaded module.
        
        Args:
            module: Python module to inspect
            
        Returns:
            List of plugin classes found in the module
        """
        plugins = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != MCPToolPlugin and
                issubclass(obj, MCPToolPlugin) and
                not inspect.isabstract(obj)):

                # Check if class is defined in this module (not imported)
                if obj.__module__ == module.__name__:
                    plugins.append(obj)
                    logger.debug(f"Found plugin class: {obj.__name__}")

        return plugins

    def auto_discover(self,
                     search_directories: list[Path] | None = None,
                     search_packages: list[str] | None = None,
                     include_builtin: bool = True) -> dict[str, int]:
        """Automatically discover and register plugins from multiple sources.
        
        Args:
            search_directories: List of directories to search
            search_packages: List of packages to search
            include_builtin: Whether to include built-in plugins
            
        Returns:
            Dictionary with discovery statistics
        """
        stats = {
            "directories_searched": 0,
            "packages_searched": 0,
            "plugins_discovered": 0,
            "plugins_registered": 0,
            "errors": 0
        }

        discovered_plugins = []

        # Discover from directories
        directories_to_search = search_directories or self._discovery_paths
        for directory in directories_to_search:
            try:
                plugins = self.discover_from_directory(directory)
                discovered_plugins.extend(plugins)
                stats["directories_searched"] += 1
            except Exception as e:
                logger.error(f"Error discovering plugins in {directory}: {e}")
                stats["errors"] += 1

        # Discover from packages
        for package_name in search_packages or []:
            try:
                plugins = self.discover_from_package(package_name)
                discovered_plugins.extend(plugins)
                stats["packages_searched"] += 1
            except Exception as e:
                logger.error(f"Error discovering plugins in package {package_name}: {e}")
                stats["errors"] += 1

        # Discover built-in plugins
        if include_builtin:
            try:
                plugins = self.discover_builtin_plugins()
                discovered_plugins.extend(plugins)
            except Exception as e:
                logger.error(f"Error discovering built-in plugins: {e}")
                stats["errors"] += 1

        stats["plugins_discovered"] = len(discovered_plugins)

        # Register discovered plugins
        for plugin_class in discovered_plugins:
            try:
                if self.registry.register_plugin_class(plugin_class):
                    stats["plugins_registered"] += 1
                else:
                    stats["errors"] += 1
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
                stats["errors"] += 1

        logger.info(f"Auto-discovery complete: {stats}")
        return stats

    def discover_and_load(self,
                         search_directories: list[Path] | None = None,
                         search_packages: list[str] | None = None,
                         include_builtin: bool = True) -> dict[str, Any]:
        """Discover plugins and immediately load them.
        
        Args:
            search_directories: List of directories to search
            search_packages: List of packages to search
            include_builtin: Whether to include built-in plugins
            
        Returns:
            Dictionary with discovery and loading statistics
        """
        # First discover plugins
        discovery_stats = self.auto_discover(
            search_directories=search_directories,
            search_packages=search_packages,
            include_builtin=include_builtin
        )

        # Then load all registered plugins
        load_stats = self.registry.load_all_plugins()

        return {
            "discovery": discovery_stats,
            "loading": {
                "plugins_loaded": sum(load_stats.values()),
                "total_plugins": len(load_stats),
                "load_success_rate": sum(load_stats.values()) / len(load_stats) * 100 if load_stats else 0,
                "load_results": load_stats
            }
        }

    def hot_reload_directory(self, directory: Path) -> dict[str, bool]:
        """Hot reload plugins from a directory (for development).
        
        Args:
            directory: Directory to reload plugins from
            
        Returns:
            Dictionary mapping plugin names to reload success status
        """
        results = {}

        # Clear discovered modules for this directory
        modules_to_remove = []
        for module_path in self._discovered_modules:
            if Path(module_path).parent == directory:
                modules_to_remove.append(module_path)

        for module_path in modules_to_remove:
            self._discovered_modules.remove(module_path)

        # Rediscover and reload
        try:
            plugins = self.discover_from_directory(directory)
            for plugin_class in plugins:
                plugin_name = plugin_class.__name__
                success = self.registry.reload_plugin(plugin_name)
                results[plugin_name] = success

        except Exception as e:
            logger.error(f"Hot reload failed for directory {directory}: {e}")

        return results

    def get_discovery_info(self) -> dict[str, Any]:
        """Get information about the discovery system.
        
        Returns:
            Dictionary with discovery system information
        """
        return {
            "discovery_paths": [str(path) for path in self._discovery_paths],
            "discovered_modules": list(self._discovered_modules),
            "total_discovery_paths": len(self._discovery_paths),
            "total_discovered_modules": len(self._discovered_modules)
        }
