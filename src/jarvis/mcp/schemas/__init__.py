"""
MCP Schema Management System.

This package provides centralized schema management for MCP tools,
including validation, templates, and standardization.
"""

from .integration import (
    SchemaIntegrator,
    auto_register_plugin_schema,
    get_schema_integrator,
    validate_plugin_input,
)
from .manager import SchemaManager, get_schema_manager
from .registry import SchemaRegistry, get_schema_registry
from .templates import (
    AnalyticsSchemaConfig,
    SchemaType,
    SearchSchemaConfig,
    UtilitySchemaConfig,
    VaultSchemaConfig,
    create_analytics_schema,
    create_graph_schema,
    create_health_schema,
    create_metrics_schema,
    create_search_schema,
    create_utility_schema,
    create_vault_schema,
)
from .validator import SchemaValidator, ValidationResult

__all__ = [
    "AnalyticsSchemaConfig",
    "SchemaIntegrator",
    "SchemaManager",
    "SchemaRegistry",
    "SchemaType",
    "SchemaValidator",
    "SearchSchemaConfig",
    "UtilitySchemaConfig",
    "ValidationResult",
    "VaultSchemaConfig",
    "auto_register_plugin_schema",
    "create_analytics_schema",
    "create_graph_schema",
    "create_health_schema",
    "create_metrics_schema",
    "create_search_schema",
    "create_utility_schema",
    "create_vault_schema",
    "get_schema_integrator",
    "get_schema_manager",
    "get_schema_registry",
    "validate_plugin_input",
]
