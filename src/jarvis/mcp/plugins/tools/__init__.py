"""
Built-in MCP Tool Plugins.

This package contains all the built-in MCP tool plugins that were converted
from the original hardcoded implementations.
"""

from typing import List, Type

from jarvis.mcp.plugins.base import MCPToolPlugin

# Import all plugin implementations
from .semantic_search import SemanticSearchPlugin
from .read_note import ReadNotePlugin
from .list_vaults import ListVaultsPlugin
from .search_vault import SearchVaultPlugin
from .search_graph import SearchGraphPlugin
from .search_combined import SearchCombinedPlugin
from .health_status import HealthStatusPlugin
from .performance_metrics import PerformanceMetricsPlugin

# Analytics plugins
from .get_vault_context import GetVaultContextPlugin
from .assess_quality import AssessQualityPlugin
from .analyze_domains import AnalyzeDomainsPlugin


def get_builtin_plugins() -> List[Type[MCPToolPlugin]]:
    """Get all built-in plugin classes.
    
    Returns:
        List of built-in plugin classes
    """
    return [
        SemanticSearchPlugin,
        ReadNotePlugin,
        ListVaultsPlugin,
        SearchVaultPlugin,
        SearchGraphPlugin,
        SearchCombinedPlugin,
        HealthStatusPlugin,
        PerformanceMetricsPlugin,
        # Analytics plugins
        GetVaultContextPlugin,
        AssessQualityPlugin,
        AnalyzeDomainsPlugin,
    ]


__all__ = [
    "get_builtin_plugins",
    "SemanticSearchPlugin",
    "ReadNotePlugin", 
    "ListVaultsPlugin",
    "SearchVaultPlugin",
    "SearchGraphPlugin",
    "SearchCombinedPlugin",
    "HealthStatusPlugin",
    "PerformanceMetricsPlugin",
    # Analytics plugins
    "GetVaultContextPlugin",
    "AssessQualityPlugin",
    "AnalyzeDomainsPlugin",
]