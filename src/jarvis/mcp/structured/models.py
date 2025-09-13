"""
Pydantic models for structured MCP tool responses.
"""

from typing import Any

from pydantic import BaseModel, Field


class CombinedSearchItem(BaseModel):
    type: str = Field(description="Result type: semantic|keyword|both")
    path: str
    vault_name: str
    unified_score: float
    semantic_score: float | None = None
    keyword_score: float | None = None
    match_type: str | None = None
    reasons: list[str] = Field(default_factory=list)


class CombinedSearchResponse(BaseModel):
    query: str
    total_results: int
    execution_time_ms: int
    results: list[CombinedSearchItem]
    analytics: dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseModel):
    center_path: str
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    metrics: dict[str, Any] = Field(default_factory=dict)


class GraphSearchResponse(BaseModel):
    query: str
    depth: int
    mode: str = Field(description="exact|keyword_fallback|fallback_semantic")
    execution_time_ms: int
    graphs: list[GraphData]
    discovered_notes: list[str] | None = None
    analytics: dict[str, Any] = Field(default_factory=dict)


class SemanticSearchItem(BaseModel):
    path: str
    vault_name: str
    similarity_score: float
    snippet: str | None = None


class SemanticSearchResponse(BaseModel):
    query: str
    total_results: int
    execution_time_ms: int
    similarity_threshold: float | None = None
    results: list[SemanticSearchItem]
    analytics: dict[str, Any] = Field(default_factory=dict)


class ReadNoteResponse(BaseModel):
    path: str
    vault_name: str | None = None
    content: str
    size_bytes: int
    last_modified: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheStatusResponse(BaseModel):
    cache_enabled: bool
    total_entries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    memory_usage_mb: float
    last_invalidation: str | None = None
    vault_stats: dict[str, Any] = Field(default_factory=dict)


class CacheInvalidateResponse(BaseModel):
    invalidated: bool
    vault: str | None = None
    entries_removed: int
    timestamp: str

