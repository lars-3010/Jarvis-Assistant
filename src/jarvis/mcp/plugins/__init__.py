"""
MCP Tool Plugin Architecture.

This module provides the plugin system for MCP tools, enabling dynamic discovery,
registration, and execution of tools in a modular fashion.
"""

from .base import MCPToolPlugin, PluginMetadata
from .registry import PluginRegistry
from .discovery import PluginDiscovery
from .tools import *

__all__ = [
    "MCPToolPlugin",
    "PluginMetadata", 
    "PluginRegistry",
    "PluginDiscovery",
]