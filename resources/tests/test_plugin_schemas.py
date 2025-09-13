import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_all_plugin_tools_expose_valid_json_schemas():
    """Smoke test: all loaded tools must provide Draft-7 valid inputSchema."""
    from jarvis.mcp.plugins.discovery import PluginDiscovery
    from jarvis.mcp.plugins.registry import PluginRegistry

    registry = PluginRegistry(container=None)
    discovery = PluginDiscovery(registry)

    stats = discovery.discover_and_load(include_builtin=True)
    assert stats["loading"]["plugins_loaded"] >= 1, "No plugins loaded for schema validation"

    tools = registry.get_tool_definitions()
    assert tools, "Registry returned no tool definitions"

    for tool in tools:
        schema = tool.inputSchema
        assert isinstance(schema, dict), f"Tool {tool.name} has non-dict schema"
        # Validate schema structure against Draft 7 meta-schema
        Draft7Validator.check_schema(schema)
        # Minimal expectations for our templates
        assert schema.get("type") == "object", f"Tool {tool.name} schema must be an object"
        assert "properties" in schema, f"Tool {tool.name} schema missing properties"


def test_plugin_sources_use_schema_helpers():
    """Static check: plugins should import schema helpers, not inline dict schemas."""
    plugins_dir = _project_root() / "src" / "jarvis" / "mcp" / "plugins" / "tools"
    assert plugins_dir.exists(), f"Missing plugins directory: {plugins_dir}"

    offenders: list[str] = []
    for py in plugins_dir.glob("*.py"):
        if py.name.startswith("__"):
            continue
        text = py.read_text(encoding="utf-8")
        imports_helper = "from jarvis.mcp.schemas" in text or "jarvis.mcp.schemas" in text
        inline_schema_dict = (
            "input_schema = {" in text or "inputSchema={" in text or "inputSchema = {" in text
        )
        if not imports_helper or inline_schema_dict:
            offenders.append(py.name)

    assert not offenders, f"Plugins must use schema helpers and avoid inline dict schemas: {offenders}"

