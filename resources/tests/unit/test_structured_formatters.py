import pytest

pytestmark = pytest.mark.unit

from jarvis.mcp.structured.formatters import (
    vault_search_to_json,
    list_vaults_to_json,
    health_status_to_json,
    performance_metrics_to_json,
    combined_search_to_json,
    graph_search_to_json,
)


def test_vault_search_to_json_basic():
    results = [
        {"path": "notes/a.md", "match_type": "name", "size": 123},
        {"path": "notes/b.md", "match_type": "content", "content_preview": "hello world"},
    ]
    payload = vault_search_to_json(
        query="test",
        results=results,
        search_content=True,
        limit=10,
        execution_time_ms=15,
    )
    assert payload["query"] == "test"
    assert payload["search_type"] == "content_and_filenames"
    assert payload["total_results"] == 2
    assert isinstance(payload["results"], list)
    assert payload["results"][0]["path"] == "notes/a.md"


def test_list_vaults_to_json_basic():
    vault_stats = {
        "default": {"note_count": 42, "latest_modified": 1_700_000_000},
        "archive": {"note_count": 5},
    }
    validation = {"default": True, "archive": False}
    model_info = {"encoder_info": {"model_name": "paraphrase", "device": "mps"}, "database_note_count": 47}
    search_stats = {"total_searches": 100, "avg_response_time_ms": 5.2}

    payload = list_vaults_to_json(vault_stats, validation, model_info, search_stats)
    assert "vaults" in payload and len(payload["vaults"]) == 2
    assert payload["search_config"]["model"] == "paraphrase"
    assert payload["search_stats"]["total_searches"] == 100


def test_health_status_to_json_passthrough():
    health = {
        "overall_status": "healthy",
        "services": {"vector": True},
        "databases": {"duckdb": True},
        "vaults": {"default": True},
        "metrics": {"system_uptime": 123},
        "timestamp": "2025-01-01T00:00:00Z",
    }
    payload = health_status_to_json(health)
    assert payload["overall_status"] == "healthy"
    assert payload["services"]["vector"] is True
    assert payload["timestamp"] == "2025-01-01T00:00:00Z"


def test_performance_metrics_to_json_grouping():
    metrics = {
        "mcp_tool_search_calls": 10,
        "system_cpu": 0.5,
        "custom_metric": 7,
    }
    payload = performance_metrics_to_json(metrics, filter_prefix=None, reset_after_read=False)
    assert payload["total_metrics"] == 3
    assert "mcp_tool_search_calls" in payload["metrics"]["mcp_tool"]
    assert "system_cpu" in payload["metrics"]["system"]
    assert "custom_metric" in payload["metrics"]["other"]


class DummyUnified:
    def __init__(self, path, vault, u, s=None, k=None, m=None, reasons=None):
        self.path = path
        self.vault_name = vault
        self.unified_score = u
        self.semantic_score = s
        self.keyword_score = k
        self.match_type = m
        self.match_reasons = reasons or []


def test_combined_search_to_json_distribution():
    unified = [
        DummyUnified("a.md", "default", 0.9, s=-2.0, k=None, m=None),
        DummyUnified("b.md", "default", 0.7, s=None, k=0.6, m="content"),
        DummyUnified("c.md", "default", 0.8, s=-3.0, k=0.5, m="filename"),
    ]
    payload = combined_search_to_json("query", unified, 12)
    assert payload["query"] == "query"
    assert payload["total_results"] == 3
    dist = payload["analytics"]["result_distribution"]
    assert dist["semantic"] == 1
    assert dist["keyword"] == 1
    assert dist["both"] == 1


def test_graph_search_to_json_shapes():
    graphs = [
        ("center.md", {"nodes": [{"id": 1, "label": "A"}], "relationships": [{"source": 1, "target": 1, "type": "LINK"}]}),
    ]
    payload = graph_search_to_json("q", 2, graphs, mode="exact", execution_time_ms=5, discovered_notes=["center.md"])
    assert payload["query"] == "q"
    assert payload["depth"] == 2
    assert payload["mode"] == "exact"
    assert payload["graphs"][0]["metrics"]["nodes"] == 1
    assert payload["graphs"][0]["metrics"]["relationships"] == 1
