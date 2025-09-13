"""
Schema Templates for MCP Tools.

This module provides pre-built schema templates for common tool types,
making it easier to create consistent schemas across plugins.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaType(Enum):
    """Common schema types."""
    SEARCH = "search"
    ANALYTICS = "analytics"
    VAULT = "vault"
    UTILITY = "utility"
    GRAPH = "graph"


@dataclass
class SearchSchemaConfig:
    """Configuration for search tool schemas."""

    query_required: bool = True
    enable_similarity_threshold: bool = True
    enable_vault_selection: bool = True
    enable_content_search: bool = False
    max_limit: int = 50
    default_limit: int = 10
    supported_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    additional_properties: dict[str, Any] | None = None


@dataclass
class AnalyticsSchemaConfig:
    """Configuration for analytics tool schemas."""

    vault_required: bool = False
    default_vault: str = "default"
    enable_caching: bool = True
    supported_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    additional_properties: dict[str, Any] | None = None


@dataclass
class VaultSchemaConfig:
    """Configuration for vault operation schemas."""

    path_required: bool = True
    enable_vault_selection: bool = True
    supported_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    additional_properties: dict[str, Any] | None = None


@dataclass
class UtilitySchemaConfig:
    """Configuration for utility tool schemas."""

    supported_formats: list[str] = field(default_factory=lambda: ["markdown", "json"])
    additional_properties: dict[str, Any] | None = None


def create_search_schema(config: SearchSchemaConfig | None = None) -> dict[str, Any]:
    """Create a standardized search tool schema.
    
    Args:
        config: Optional configuration for customization
        
    Returns:
        JSON schema for search tools
    """
    if config is None:
        config = SearchSchemaConfig()

    properties = {
        "query": {
            "type": "string",
            "description": "Search query",
            "minLength": 1
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results",
            "minimum": 1,
            "maximum": config.max_limit,
            "default": config.default_limit
        },
        "format": {
            "type": "string",
            "enum": config.supported_formats,
            "default": config.supported_formats[0],
            "description": "Response format"
        }
    }

    required = []
    if config.query_required:
        required.append("query")

    # Add optional properties
    if config.enable_vault_selection:
        properties["vault"] = {
            "type": "string",
            "description": "Optional vault name to search within"
        }

    if config.enable_similarity_threshold:
        properties["similarity_threshold"] = {
            "type": "number",
            "description": "Minimum similarity score (0.0-1.0)",
            "minimum": 0.0,
            "maximum": 1.0
        }

    if config.enable_content_search:
        properties["search_content"] = {
            "type": "boolean",
            "description": "Whether to search file content in addition to names",
            "default": True
        }

    # Add any additional properties
    if config.additional_properties:
        properties.update(config.additional_properties)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def create_analytics_schema(config: AnalyticsSchemaConfig | None = None) -> dict[str, Any]:
    """Create a standardized analytics tool schema.
    
    Args:
        config: Optional configuration for customization
        
    Returns:
        JSON schema for analytics tools
    """
    if config is None:
        config = AnalyticsSchemaConfig()

    properties = {
        "vault": {
            "type": "string",
            "description": "Vault name to analyze",
            "default": config.default_vault
        },
        "format": {
            "type": "string",
            "enum": config.supported_formats,
            "default": config.supported_formats[0],
            "description": "Response format"
        }
    }

    required = []
    if config.vault_required:
        required.append("vault")

    if config.enable_caching:
        properties["cache"] = {
            "type": "boolean",
            "description": "Whether to use cached results",
            "default": True
        }

    # Add any additional properties
    if config.additional_properties:
        properties.update(config.additional_properties)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def create_vault_schema(config: VaultSchemaConfig | None = None) -> dict[str, Any]:
    """Create a standardized vault operation schema.
    
    Args:
        config: Optional configuration for customization
        
    Returns:
        JSON schema for vault operations
    """
    if config is None:
        config = VaultSchemaConfig()

    properties = {
        "path": {
            "type": "string",
            "description": "File or directory path within the vault",
            "minLength": 1
        },
        "format": {
            "type": "string",
            "enum": config.supported_formats,
            "default": config.supported_formats[0],
            "description": "Response format"
        }
    }

    required = []
    if config.path_required:
        required.append("path")

    if config.enable_vault_selection:
        properties["vault"] = {
            "type": "string",
            "description": "Optional vault name"
        }

    # Add any additional properties
    if config.additional_properties:
        properties.update(config.additional_properties)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def create_utility_schema(config: UtilitySchemaConfig | None = None) -> dict[str, Any]:
    """Create a standardized utility tool schema.
    
    Args:
        config: Optional configuration for customization
        
    Returns:
        JSON schema for utility tools
    """
    if config is None:
        config = UtilitySchemaConfig()

    properties = {
        "format": {
            "type": "string",
            "enum": config.supported_formats,
            "default": config.supported_formats[0],
            "description": "Response format"
        }
    }

    # Add any additional properties
    if config.additional_properties:
        properties.update(config.additional_properties)

    return {
        "type": "object",
        "properties": properties,
        "required": []
    }


def create_graph_schema(
    enable_depth_control: bool = True,
    max_depth: int = 3,
    additional_properties: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a standardized graph operation schema.
    
    Args:
        enable_depth_control: Whether to include depth parameter
        max_depth: Maximum traversal depth
        additional_properties: Additional properties to include
        
    Returns:
        JSON schema for graph operations
    """
    properties = {
        "query_note_path": {
            "type": "string",
            "description": "Path to the note to start graph traversal from",
            "minLength": 1
        },
        "format": {
            "type": "string",
            "enum": ["markdown", "json"],
            "default": "markdown",
            "description": "Response format"
        }
    }

    required = ["query_note_path"]

    if enable_depth_control:
        properties["depth"] = {
            "type": "integer",
            "description": "Graph traversal depth",
            "minimum": 1,
            "maximum": max_depth,
            "default": 1
        }

    # Add any additional properties
    if additional_properties:
        properties.update(additional_properties)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def create_health_schema() -> dict[str, Any]:
    """Create a standardized health check schema.
    
    Returns:
        JSON schema for health check tools
    """
    return {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "default": "json",
                "description": "Response format"
            },
            "include_details": {
                "type": "boolean",
                "description": "Whether to include detailed health information",
                "default": False
            }
        },
        "required": []
    }


def create_metrics_schema() -> dict[str, Any]:
    """Create a standardized metrics schema.
    
    Returns:
        JSON schema for metrics tools
    """
    return {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["json", "markdown"],
                "default": "json",
                "description": "Response format"
            },
            "reset_after_read": {
                "type": "boolean",
                "description": "Whether to reset metrics after reading",
                "default": False
            },
            "filter_prefix": {
                "type": "string",
                "description": "Filter metrics by prefix",
                "minLength": 1
            }
        },
        "required": []
    }


def merge_schemas(*schemas: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple schemas into a single schema.
    
    Args:
        *schemas: Schemas to merge
        
    Returns:
        Merged schema
    """
    if not schemas:
        return {"type": "object", "properties": {}, "required": []}

    merged_properties = {}
    merged_required = []

    for schema in schemas:
        if isinstance(schema, dict):
            # Merge properties
            schema_props = schema.get("properties", {})
            merged_properties.update(schema_props)

            # Merge required fields
            schema_required = schema.get("required", [])
            merged_required.extend(schema_required)

    # Remove duplicates from required
    merged_required = list(set(merged_required))

    return {
        "type": "object",
        "properties": merged_properties,
        "required": merged_required
    }


def extend_schema(
    base_schema: dict[str, Any],
    additional_properties: dict[str, Any],
    additional_required: list[str] | None = None
) -> dict[str, Any]:
    """Extend a base schema with additional properties.
    
    Args:
        base_schema: Base schema to extend
        additional_properties: Properties to add
        additional_required: Additional required fields
        
    Returns:
        Extended schema
    """
    extended = base_schema.copy()

    # Extend properties
    if "properties" not in extended:
        extended["properties"] = {}
    extended["properties"].update(additional_properties)

    # Extend required fields
    if additional_required:
        existing_required = extended.get("required", [])
        extended["required"] = list(set(existing_required + additional_required))

    return extended


def create_schema_from_template(
    template_type: SchemaType,
    **kwargs: Any
) -> dict[str, Any]:
    """Create a schema from a template type.
    
    Args:
        template_type: Type of schema template to use
        **kwargs: Configuration arguments for the template
        
    Returns:
        Generated schema
    """
    if template_type == SchemaType.SEARCH:
        config = SearchSchemaConfig(**kwargs) if kwargs else None
        return create_search_schema(config)
    elif template_type == SchemaType.ANALYTICS:
        config = AnalyticsSchemaConfig(**kwargs) if kwargs else None
        return create_analytics_schema(config)
    elif template_type == SchemaType.VAULT:
        config = VaultSchemaConfig(**kwargs) if kwargs else None
        return create_vault_schema(config)
    elif template_type == SchemaType.UTILITY:
        config = UtilitySchemaConfig(**kwargs) if kwargs else None
        return create_utility_schema(config)
    elif template_type == SchemaType.GRAPH:
        return create_graph_schema(**kwargs)
    else:
        raise ValueError(f"Unknown schema template type: {template_type}")


def get_template_config_class(template_type: SchemaType) -> type:
    """Get the configuration class for a template type.
    
    Args:
        template_type: Schema template type
        
    Returns:
        Configuration class for the template
    """
    config_map = {
        SchemaType.SEARCH: SearchSchemaConfig,
        SchemaType.ANALYTICS: AnalyticsSchemaConfig,
        SchemaType.VAULT: VaultSchemaConfig,
        SchemaType.UTILITY: UtilitySchemaConfig,
    }

    return config_map.get(template_type, dict)
