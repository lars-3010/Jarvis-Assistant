import pytest

from jarvis.mcp.plugins.registry import PluginRegistry
from jarvis.mcp.plugins.tools import get_builtin_plugins
from jarvis.mcp.schemas.registry import get_schema_registry, reset_schema_registry


@pytest.mark.unit
def test_registry_loads_and_schemas_validate():
    # Fresh schema registry for test isolation
    reset_schema_registry()

    registry = PluginRegistry(container=None)
    # Register builtin plugin classes
    for cls in get_builtin_plugins():
        registry.register_plugin_class(cls)

    # Load instances (auto-registers schemas)
    load_results = registry.load_all_plugins()
    assert any(load_results.values()), "Expected at least some plugins to load"

    # Validate that schema registry holds schemas and performs input validation
    schema_registry = get_schema_registry()

    # search-semantic should require 'query'
    validation = schema_registry.validate_tool_input("search-semantic", {})
    assert not validation.is_valid

    # get-health-status accepts empty input; schema exists
    validation2 = schema_registry.validate_tool_input("get-health-status", {})
    assert validation2 is not None and validation2.is_valid

    # search-graph requires 'query_note_path'
    validation3 = schema_registry.validate_tool_input("search-graph", {})
    assert not validation3.is_valid
