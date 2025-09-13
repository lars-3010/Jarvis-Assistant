"""
Schema Integration for MCP Plugin System.

This module provides integration between the schema management system
and the MCP plugin registry, enabling automatic schema registration
and validation for all plugins.
"""

from typing import Any

from jarvis.mcp.plugins.base import MCPToolPlugin
from jarvis.utils.logging import setup_logging

from .manager import get_schema_manager
from .registry import get_schema_registry
from .templates import (
    AnalyticsSchemaConfig,
    SchemaType,
    SearchSchemaConfig,
    UtilitySchemaConfig,
    VaultSchemaConfig,
    create_analytics_schema,
    create_graph_schema,
    create_search_schema,
    create_utility_schema,
    create_vault_schema,
)

logger = setup_logging(__name__)


class SchemaIntegrator:
    """Integrates schema management with plugin system."""

    def __init__(self):
        """Initialize schema integrator."""
        self.schema_manager = get_schema_manager()
        self._registered_plugins: dict[str, str] = {}

        logger.info("Schema integrator initialized")

    def register_plugin_schema(self, plugin: MCPToolPlugin) -> bool:
        """Register schema for a plugin automatically.
        
        Args:
            plugin: Plugin instance to register schema for
            
        Returns:
            True if registration successful
        """
        plugin_name = plugin.name

        # Always (re)register schema to keep registry state authoritative and fresh

        try:
            # Get schema from plugin
            tool_def = plugin.get_tool_definition()
            if not tool_def or not hasattr(tool_def, 'inputSchema'):
                logger.warning(f"Plugin {plugin_name} has no input schema")
                return False

            schema = tool_def.inputSchema
            if not schema:
                logger.warning(f"Plugin {plugin_name} has empty input schema")
                return False

            # Determine category based on plugin type and tags
            category = self._infer_plugin_category(plugin)

            # Register with schema registry (always re-register; idempotent at manager level)
            # Fetch the current (possibly reset) schema registry
            from .registry import get_schema_registry as _get_schema_registry
            schema_registry = _get_schema_registry(self.schema_manager)

            success = schema_registry.register_tool_schema(
                tool_name=plugin_name,
                schema=schema,
                description=plugin.description,
                category=category,
                version=plugin.version
            )

            if success:
                self._registered_plugins[plugin_name] = category
                logger.info(f"(Re)registered schema for plugin: {plugin_name} (category: {category})")

            return success

        except Exception as e:
            logger.error(f"Failed to register schema for plugin {plugin_name}: {e}")
            return False

    def validate_plugin_input(self, plugin_name: str, input_data: Any) -> bool:
        """Validate input data for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            input_data: Input data to validate
            
        Returns:
            True if validation passes
        """
        from .registry import get_schema_registry as _get_schema_registry
        schema_registry = _get_schema_registry(self.schema_manager)
        validation_result = schema_registry.validate_tool_input(plugin_name, input_data)

        if not validation_result.is_valid:
            logger.error(f"Input validation failed for {plugin_name}:")
            for error in validation_result.errors:
                logger.error(f"  {error.path}: {error.message}")
            return False

        if validation_result.has_warnings:
            logger.warning(f"Input validation warnings for {plugin_name}:")
            for warning in validation_result.warnings:
                logger.warning(f"  {warning.path}: {warning.message}")

        return True

    def get_standardized_schema(self, schema_type: SchemaType, **config_kwargs) -> dict[str, Any]:
        """Get a standardized schema for a given type.
        
        Args:
            schema_type: Type of schema to generate
            **config_kwargs: Configuration parameters for the schema
            
        Returns:
            Generated schema definition
        """
        if schema_type == SchemaType.SEARCH:
            config = SearchSchemaConfig(**config_kwargs) if config_kwargs else None
            return create_search_schema(config)
        elif schema_type == SchemaType.ANALYTICS:
            config = AnalyticsSchemaConfig(**config_kwargs) if config_kwargs else None
            return create_analytics_schema(config)
        elif schema_type == SchemaType.VAULT:
            config = VaultSchemaConfig(**config_kwargs) if config_kwargs else None
            return create_vault_schema(config)
        elif schema_type == SchemaType.UTILITY:
            config = UtilitySchemaConfig(**config_kwargs) if config_kwargs else None
            return create_utility_schema(config)
        elif schema_type == SchemaType.GRAPH:
            return create_graph_schema(**config_kwargs)
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

    def suggest_schema_improvements(self, plugin: MCPToolPlugin) -> list[str]:
        """Suggest schema improvements for a plugin.
        
        Args:
            plugin: Plugin to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []

        try:
            # Get current schema
            tool_def = plugin.get_tool_definition()
            if not tool_def or not hasattr(tool_def, 'inputSchema'):
                return ["Plugin has no input schema defined"]

            current_schema = tool_def.inputSchema
            plugin_category = self._infer_plugin_category(plugin)

            # Compare with standardized schema
            if plugin_category == "search":
                standard_schema = create_search_schema()
                suggestions.extend(self._compare_schemas(current_schema, standard_schema, "search"))
            elif plugin_category == "analytics":
                standard_schema = create_analytics_schema()
                suggestions.extend(self._compare_schemas(current_schema, standard_schema, "analytics"))
            elif plugin_category == "vault":
                standard_schema = create_vault_schema()
                suggestions.extend(self._compare_schemas(current_schema, standard_schema, "vault"))
            elif plugin_category == "utility":
                standard_schema = create_utility_schema()
                suggestions.extend(self._compare_schemas(current_schema, standard_schema, "utility"))

            # General suggestions
            properties = current_schema.get("properties", {})

            # Check for common missing properties
            if "format" not in properties:
                suggestions.append("Consider adding a 'format' parameter for consistent output formatting")

            # Check for description completeness
            missing_descriptions = []
            for prop_name, prop_def in properties.items():
                if not prop_def.get("description"):
                    missing_descriptions.append(prop_name)

            if missing_descriptions:
                suggestions.append(f"Add descriptions for properties: {', '.join(missing_descriptions)}")

        except Exception as e:
            logger.error(f"Failed to analyze schema for {plugin.name}: {e}")
            suggestions.append(f"Schema analysis failed: {e}")

        return suggestions

    def get_plugin_schema_stats(self) -> dict[str, Any]:
        """Get statistics about registered plugin schemas.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_registered": len(self._registered_plugins),
            "by_category": self._count_by_category(),
            "registry_stats": self.schema_registry.get_registry_stats()
        }

    def _infer_plugin_category(self, plugin: MCPToolPlugin) -> str:
        """Infer category from plugin type and tags.
        
        Args:
            plugin: Plugin to analyze
            
        Returns:
            Inferred category
        """
        # Check plugin class hierarchy
        class_name = plugin.__class__.__name__.lower()
        if "search" in class_name:
            return "search"
        elif "analytic" in class_name:
            return "analytics"
        elif "vault" in class_name:
            return "vault"
        elif "graph" in class_name:
            return "graph"
        elif any(word in class_name for word in ["health", "metric", "performance"]):
            return "utility"

        # Check plugin tags
        tags = plugin.tags
        if any(tag in ["search", "semantic", "query"] for tag in tags):
            return "search"
        elif any(tag in ["analytics", "analysis"] for tag in tags):
            return "analytics"
        elif any(tag in ["vault", "files"] for tag in tags):
            return "vault"
        elif any(tag in ["graph", "relationships", "network"] for tag in tags):
            return "graph"
        elif any(tag in ["health", "metrics", "monitoring", "utility"] for tag in tags):
            return "utility"

        return "general"

    def _compare_schemas(self, current: dict[str, Any], standard: dict[str, Any], category: str) -> list[str]:
        """Compare current schema with standard schema.
        
        Args:
            current: Current schema definition
            standard: Standard schema definition
            category: Schema category
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []

        current_props = current.get("properties", {})
        standard_props = standard.get("properties", {})

        # Check for missing standard properties
        for prop_name, prop_def in standard_props.items():
            if prop_name not in current_props:
                suggestions.append(f"Consider adding standard '{prop_name}' property for {category} tools")

        # Check for different property types
        for prop_name in current_props:
            if prop_name in standard_props:
                current_type = current_props[prop_name].get("type")
                standard_type = standard_props[prop_name].get("type")
                if current_type != standard_type:
                    suggestions.append(f"Property '{prop_name}' type differs from standard (current: {current_type}, standard: {standard_type})")

        return suggestions

    def _count_by_category(self) -> dict[str, int]:
        """Count registered plugins by category.
        
        Returns:
            Category counts
        """
        category_counts = {}
        for category in self._registered_plugins.values():
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts


# Global integrator instance
_global_integrator: SchemaIntegrator | None = None


def get_schema_integrator() -> SchemaIntegrator:
    """Get the global schema integrator instance.
    
    Returns:
        Global schema integrator instance
    """
    global _global_integrator
    if _global_integrator is None:
        _global_integrator = SchemaIntegrator()
    return _global_integrator


def auto_register_plugin_schema(plugin: MCPToolPlugin) -> bool:
    """Automatically register schema for a plugin.
    
    Args:
        plugin: Plugin instance
        
    Returns:
        True if registration successful
    """
    integrator = get_schema_integrator()
    return integrator.register_plugin_schema(plugin)


def validate_plugin_input(plugin_name: str, input_data: Any) -> bool:
    """Validate input data for a plugin.
    
    Args:
        plugin_name: Name of the plugin
        input_data: Input data to validate
        
    Returns:
        True if validation passes
    """
    integrator = get_schema_integrator()
    return integrator.validate_plugin_input(plugin_name, input_data)
