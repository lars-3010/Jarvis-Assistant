"""
Built-in MCP Tool Plugins.

This package contains all the built-in MCP tool plugins that were converted
from the original hardcoded implementations.
"""

from typing import List, Type

from jarvis.mcp.plugins.base import MCPToolPlugin

from .analytics_cache_status import AnalyticsCacheStatusPlugin
from .analytics_invalidate_cache import AnalyticsInvalidateCachePlugin
from .analyze_domains import AnalyzeDomainsPlugin
from .assess_quality import AssessQualityPlugin

# Analytics plugins
from .get_vault_context import GetVaultContextPlugin
from .health_status import HealthStatusPlugin
from .list_vaults import ListVaultsPlugin
from .performance_metrics import PerformanceMetricsPlugin
from .read_note import ReadNotePlugin
from .search_combined import SearchCombinedPlugin
from .search_graph import SearchGraphPlugin
from .search_graphrag import SearchGraphRAGPlugin
from .search_vault import SearchVaultPlugin

# Import all plugin implementations
from .semantic_search import SemanticSearchPlugin


def get_builtin_plugins() -> list[type[MCPToolPlugin]]:
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
        AnalyticsCacheStatusPlugin,
        AnalyticsInvalidateCachePlugin,
        SearchGraphRAGPlugin,
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
    "AnalyticsCacheStatusPlugin",
    "AnalyticsInvalidateCachePlugin",
    "SearchGraphRAGPlugin",
]
