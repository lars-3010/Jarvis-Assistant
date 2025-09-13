"""
Helpers to convert internal search results to structured JSON payloads.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

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

SCHEMA_VERSION = "v1"


def combined_search_to_json(
    query: str,
    unified_results: Sequence[Any],
    execution_time_ms: int,
) -> dict[str, Any]:
    items: list[CombinedSearchItem] = []
    counts = {"semantic": 0, "keyword": 0, "both": 0}

    for r in unified_results:
        has_sem = getattr(r, "semantic_score", None) is not None
        has_key = getattr(r, "keyword_score", None) is not None
        if has_sem and has_key:
            rtype = "both"
        elif has_sem:
            rtype = "semantic"
        else:
            rtype = "keyword"
        counts[rtype] = counts.get(rtype, 0) + 1

        items.append(
            CombinedSearchItem(
                type=rtype,
                path=str(getattr(r, "path", "")),
                vault_name=str(getattr(r, "vault_name", "")),
                unified_score=float(getattr(r, "unified_score", 0.0)),
                semantic_score=getattr(r, "semantic_score", None),
                keyword_score=getattr(r, "keyword_score", None),
                match_type=getattr(r, "match_type", None),
                reasons=list(getattr(r, "match_reasons", []) or []),
            )
        )

    resp = CombinedSearchResponse(
        query=query,
        total_results=len(items),
        execution_time_ms=execution_time_ms,
        results=items,
        analytics={
            "result_distribution": counts,
        },
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def graph_search_to_json(
    query: str,
    depth: int,
    graphs: Sequence[tuple[str, dict[str, Any]]],
    mode: str,
    execution_time_ms: int,
    discovered_notes: list[str] | None = None,
) -> dict[str, Any]:
    graph_items: list[GraphData] = []

    for center_path, g in graphs:
        nodes = g.get("nodes", []) if isinstance(g, dict) else []
        rels = g.get("relationships", []) if isinstance(g, dict) else []
        metrics = {
            "nodes": len(nodes),
            "relationships": len(rels),
            "relationship_types": len({
                rel.get("original_type", rel.get("type", "UNKNOWN"))
                for rel in rels
                if isinstance(rel, dict)
            }),
        }
        graph_items.append(
            GraphData(
                center_path=center_path,
                nodes=nodes,
                relationships=rels,
                metrics=metrics,
            )
        )

    resp = GraphSearchResponse(
        query=query,
        depth=depth,
        mode=mode,
        execution_time_ms=execution_time_ms,
        graphs=graph_items,
        discovered_notes=discovered_notes,
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def semantic_fallback_to_json(
    query: str,
    semantic_results: Sequence[Any],
    execution_time_ms: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for res in semantic_results or []:
        items.append(
            {
                "path": str(getattr(res, "path", "")),
                "vault_name": str(getattr(res, "vault_name", "")),
                "score": float(getattr(res, "similarity_score", 0.0)),
            }
        )
    return {
        "query": query,
        "mode": "fallback_semantic",
        "execution_time_ms": execution_time_ms,
        "results": items,
        "schema_version": SCHEMA_VERSION,
    }


def _safe_preview(text: str, limit: int = 100) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def vault_search_to_json(
    query: str,
    results: Sequence[Any],
    search_content: bool,
    limit: int,
    execution_time_ms: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for r in results or []:
        if isinstance(r, dict):
            path = r.get("path", "Unknown")
            match_type = r.get("match_type", "name")
            size = r.get("size")
            content_preview = _safe_preview(r.get("content_preview", "") or "")
        else:
            path = str(getattr(r, "path", "Unknown"))
            match_type = getattr(r, "match_type", "name")
            size = getattr(r, "size", None)
            content_preview = _safe_preview(getattr(r, "content_preview", None) or "")
        item: dict[str, Any] = {
            "path": path,
            "match_type": match_type,
        }
        if size is not None:
            item["size_bytes"] = size
        if content_preview:
            item["preview"] = content_preview
        items.append(item)

    return {
        "query": query,
        "search_type": "content_and_filenames" if search_content else "filenames",
        "limit": limit,
        "total_results": len(items),
        "execution_time_ms": execution_time_ms,
        "results": items,
        "schema_version": SCHEMA_VERSION,
    }


def list_vaults_to_json(
    vault_stats: dict[str, Any],
    validation: dict[str, bool],
    model_info: dict[str, Any] | None,
    search_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    vaults: list[dict[str, Any]] = []
    for name, stats in (vault_stats or {}).items():
        entry: dict[str, Any] = {
            "name": name,
            "is_available": bool(validation.get(name, False)),
            "note_count": int(stats.get("note_count", 0)),
        }
        latest = stats.get("latest_modified")
        if latest:
            try:
                entry["latest_modified_iso"] = datetime.fromtimestamp(latest).isoformat()
            except Exception:
                entry["latest_modified"] = latest
        vaults.append(entry)

    encoder_info = (model_info or {}).get("encoder_info", {}) if model_info else {}
    search_cfg = {
        "model": encoder_info.get("model_name", "Unknown"),
        "device": encoder_info.get("device", "Unknown"),
        "database_note_count": (model_info or {}).get("database_note_count", 0),
    }

    return {
        "vaults": vaults,
        "search_config": search_cfg,
        "search_stats": search_stats or {},
        "schema_version": SCHEMA_VERSION,
    }


def health_status_to_json(health_status: dict[str, Any]) -> dict[str, Any]:
    overall = health_status.get("overall_status", "unknown")
    return {
        "overall_status": overall,
        "services": health_status.get("services", {}),
        "databases": health_status.get("databases", {}),
        "vaults": health_status.get("vaults", {}),
        "metrics": health_status.get("metrics", {}),
        "timestamp": health_status.get("timestamp"),
        "schema_version": SCHEMA_VERSION,
    }


def performance_metrics_to_json(
    all_metrics: dict[str, Any],
    filter_prefix: str | None,
    reset_after_read: bool,
) -> dict[str, Any]:
    filtered = {
        k: v for k, v in all_metrics.items()
        if not filter_prefix or k.startswith(filter_prefix)
    }
    grouped = {
        "mcp_tool": {k: v for k, v in filtered.items() if k.startswith("mcp_tool_")},
        "system": {k: v for k, v in filtered.items() if k.startswith("system_") or k.startswith("service_")},
        "other": {k: v for k, v in filtered.items() if not (k.startswith("mcp_tool_") or k.startswith("system_") or k.startswith("service_"))},
    }
    return {
        "filter_prefix": filter_prefix,
        "reset_after_read": reset_after_read,
        "total_metrics": len(filtered),
        "metrics": grouped,
        "schema_version": SCHEMA_VERSION,
    }


def graphrag_to_json(
    query: str,
    sources: list[dict[str, Any]],
    graphs: list[GraphData],
    execution_time_ms: int,
    limits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": query,
        "sources": sources,
        "graphs": [g.model_dump() for g in graphs],
        "analytics": {
            "execution_time_ms": execution_time_ms,
            "limits": limits,
        },
        "schema_version": SCHEMA_VERSION,
    }


def semantic_search_to_json(
    query: str,
    results: Sequence[Any],
    execution_time_ms: int,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    items: list[SemanticSearchItem] = []
    for r in results or []:
        # Handle both object and dict formats
        if isinstance(r, dict):
            path = r.get("path", "Unknown")
            vault_name = r.get("vault_name", "")
            score = float(r.get("similarity_score", 0.0))
            snippet = r.get("snippet") or r.get("context_snippet")
        else:
            path = str(getattr(r, "path", "Unknown"))
            vault_name = str(getattr(r, "vault_name", ""))
            score = float(getattr(r, "similarity_score", 0.0))
            snippet = getattr(r, "snippet", None) or getattr(r, "context_snippet", None)

        items.append(
            SemanticSearchItem(
                path=path,
                vault_name=vault_name,
                similarity_score=score,
                snippet=_safe_preview(snippet) if snippet else None,
            )
        )

    resp = SemanticSearchResponse(
        query=query,
        total_results=len(items),
        execution_time_ms=execution_time_ms,
        similarity_threshold=similarity_threshold,
        results=items,
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def read_note_to_json(
    path: str,
    content: str,
    vault_name: str | None = None,
    size_bytes: int | None = None,
    last_modified: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = ReadNoteResponse(
        path=path,
        vault_name=vault_name,
        content=content,
        size_bytes=size_bytes or len(content.encode('utf-8')),
        last_modified=last_modified,
        metadata=metadata or {},
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def cache_status_to_json(
    cache_enabled: bool,
    stats: dict[str, Any],
    vault_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_entries = stats.get("total_entries", 0)
    cache_hits = stats.get("cache_hits", 0)
    cache_misses = stats.get("cache_misses", 0)
    hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0

    resp = CacheStatusResponse(
        cache_enabled=cache_enabled,
        total_entries=total_entries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        memory_usage_mb=stats.get("memory_usage_mb", 0.0),
        last_invalidation=stats.get("last_invalidation"),
        vault_stats=vault_stats or {},
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def cache_invalidate_to_json(
    invalidated: bool,
    vault: str | None = None,
    entries_removed: int = 0,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if not timestamp:
        timestamp = datetime.now().isoformat()

    resp = CacheInvalidateResponse(
        invalidated=invalidated,
        vault=vault,
        entries_removed=entries_removed,
        timestamp=timestamp,
    )
    payload = resp.model_dump()
    payload["schema_version"] = SCHEMA_VERSION
    return payload
