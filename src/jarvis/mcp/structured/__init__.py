"""
Structured response scaffolding for MCP tools.

This package provides shared models and formatting helpers to emit
consistent JSON responses across tools while keeping markdown as
the default human-friendly format.
"""

from .formatters import (
    cache_invalidate_to_json,
    cache_status_to_json,
    combined_search_to_json,
    graph_search_to_json,
    graphrag_to_json,
    health_status_to_json,
    list_vaults_to_json,
    performance_metrics_to_json,
    read_note_to_json,
    semantic_fallback_to_json,
    semantic_search_to_json,
    vault_search_to_json,
)
from .models import (
    CacheInvalidateResponse,
    CacheStatusResponse,
    CombinedSearchItem,
    CombinedSearchResponse,
    GraphData,
    GraphSearchResponse,
    ReadNoteResponse,
    SemanticSearchItem,
    SemanticSearchResponse,
)

__all__ = [
    # Models
    "CacheInvalidateResponse",
    "CacheStatusResponse",
    "CombinedSearchItem",
    "CombinedSearchResponse",
    "GraphData",
    "GraphSearchResponse",
    "ReadNoteResponse",
    "SemanticSearchItem",
    "SemanticSearchResponse",
    # Helpers
    "cache_invalidate_to_json",
    "cache_status_to_json",
    "combined_search_to_json",
    "graph_search_to_json",
    "graphrag_to_json",
    "health_status_to_json",
    "list_vaults_to_json",
    "performance_metrics_to_json",
    "read_note_to_json",
    "semantic_fallback_to_json",
    "semantic_search_to_json",
    "vault_search_to_json",
]
