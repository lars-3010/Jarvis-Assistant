"""
Central Schema Manager for MCP Tools.

This module provides centralized management of JSON schemas for all MCP tools,
including validation, registration, and template management.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class SchemaError(JarvisError):
    """Schema-related error."""
    pass


@dataclass
class SchemaInfo:
    """Information about a registered schema."""

    name: str
    version: str
    schema: dict[str, Any]
    description: str | None = None
    category: str | None = None
    tags: list[str] | None = None
    created_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SchemaManager:
    """Central manager for MCP tool schemas."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize schema manager.
        
        Args:
            config: Optional configuration for schema management
        """
        self.config = config or {}
        self._schemas: dict[str, SchemaInfo] = {}
        self._validators: dict[str, Draft7Validator] = {}
        self._schema_categories: dict[str, list[str]] = {}

        # Load built-in schemas
        self._load_builtin_schemas()

        logger.info("Schema manager initialized")

    def register_schema(
        self,
        name: str,
        schema: dict[str, Any],
        version: str = "1.0.0",
        description: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        created_by: str | None = None,
        validate: bool = True
    ) -> bool:
        """Register a new schema.
        
        Args:
            name: Unique name for the schema
            schema: JSON schema definition
            version: Schema version
            description: Optional description
            category: Optional category
            tags: Optional tags for organization
            created_by: Optional creator identifier
            validate: Whether to validate the schema itself
            
        Returns:
            True if registration successful
        """
        try:
            # Validate the schema structure if requested
            if validate:
                Draft7Validator.check_schema(schema)

            # Check for existing schema
            if name in self._schemas:
                logger.warning(f"Overwriting existing schema: {name}")

            # Create schema info
            schema_info = SchemaInfo(
                name=name,
                version=version,
                schema=schema,
                description=description,
                category=category,
                tags=tags or [],
                created_by=created_by
            )

            # Register schema and create validator
            self._schemas[name] = schema_info
            self._validators[name] = Draft7Validator(schema)

            # Update category index
            if category:
                if category not in self._schema_categories:
                    self._schema_categories[category] = []
                self._schema_categories[category].append(name)

            logger.info(f"Registered schema: {name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Failed to register schema {name}: {e}")
            return False

    def get_schema(self, name: str) -> SchemaInfo | None:
        """Get schema information by name.
        
        Args:
            name: Schema name
            
        Returns:
            Schema information or None if not found
        """
        return self._schemas.get(name)

    def get_schema_definition(self, name: str) -> dict[str, Any] | None:
        """Get raw schema definition by name.
        
        Args:
            name: Schema name
            
        Returns:
            Schema definition or None if not found
        """
        schema_info = self._schemas.get(name)
        return schema_info.schema if schema_info else None

    def validate_data(self, schema_name: str, data: Any) -> list[str]:
        """Validate data against a registered schema.
        
        Args:
            schema_name: Name of the schema to validate against
            data: Data to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        if schema_name not in self._validators:
            return [f"Schema '{schema_name}' not found"]

        validator = self._validators[schema_name]
        errors = []

        try:
            for error in validator.iter_errors(data):
                error_path = ".".join(str(p) for p in error.path) if error.path else "root"
                errors.append(f"{error_path}: {error.message}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        return errors

    def is_valid(self, schema_name: str, data: Any) -> bool:
        """Check if data is valid against a schema.
        
        Args:
            schema_name: Name of the schema to validate against
            data: Data to validate
            
        Returns:
            True if data is valid
        """
        return len(self.validate_data(schema_name, data)) == 0

    def list_schemas(self, category: str | None = None) -> list[str]:
        """List registered schema names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of schema names
        """
        if category:
            return self._schema_categories.get(category, [])
        return list(self._schemas.keys())

    def get_schema_categories(self) -> list[str]:
        """Get all schema categories.
        
        Returns:
            List of category names
        """
        return list(self._schema_categories.keys())

    def search_schemas(
        self,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None
    ) -> list[SchemaInfo]:
        """Search schemas by criteria.
        
        Args:
            query: Optional text query for name/description
            category: Optional category filter
            tags: Optional tag filters
            
        Returns:
            List of matching schema info objects
        """
        results = []

        for schema_info in self._schemas.values():
            # Category filter
            if category and schema_info.category != category:
                continue

            # Tag filter
            if tags and not any(tag in (schema_info.tags or []) for tag in tags):
                continue

            # Query filter
            if query:
                query_lower = query.lower()
                if not any(query_lower in text.lower() for text in [
                    schema_info.name,
                    schema_info.description or "",
                    " ".join(schema_info.tags or [])
                ]):
                    continue

            results.append(schema_info)

        return results

    def unregister_schema(self, name: str) -> bool:
        """Unregister a schema.
        
        Args:
            name: Schema name to unregister
            
        Returns:
            True if unregistration successful
        """
        if name not in self._schemas:
            logger.warning(f"Schema not found for unregistration: {name}")
            return False

        try:
            schema_info = self._schemas[name]

            # Remove from main registry
            del self._schemas[name]
            del self._validators[name]

            # Remove from category index
            if schema_info.category and schema_info.category in self._schema_categories:
                if name in self._schema_categories[schema_info.category]:
                    self._schema_categories[schema_info.category].remove(name)

                # Clean up empty categories
                if not self._schema_categories[schema_info.category]:
                    del self._schema_categories[schema_info.category]

            logger.info(f"Unregistered schema: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister schema {name}: {e}")
            return False

    def export_schema(self, name: str, file_path: Path | None = None) -> str | None:
        """Export a schema to JSON file or string.
        
        Args:
            name: Schema name to export
            file_path: Optional file path to save to
            
        Returns:
            JSON string if file_path is None, otherwise None
        """
        schema_info = self._schemas.get(name)
        if not schema_info:
            logger.error(f"Schema not found for export: {name}")
            return None

        try:
            export_data = schema_info.to_dict()
            json_str = json.dumps(export_data, indent=2)

            if file_path:
                file_path.write_text(json_str)
                logger.info(f"Exported schema {name} to {file_path}")
                return None
            else:
                return json_str

        except Exception as e:
            logger.error(f"Failed to export schema {name}: {e}")
            return None

    def import_schema(self, file_path: Path) -> bool:
        """Import a schema from JSON file.
        
        Args:
            file_path: Path to JSON schema file
            
        Returns:
            True if import successful
        """
        try:
            data = json.loads(file_path.read_text())

            # Extract schema info
            name = data.get("name")
            if not name:
                logger.error(f"Schema file missing name field: {file_path}")
                return False

            schema = data.get("schema")
            if not schema:
                logger.error(f"Schema file missing schema field: {file_path}")
                return False

            # Register the schema
            return self.register_schema(
                name=name,
                schema=schema,
                version=data.get("version", "1.0.0"),
                description=data.get("description"),
                category=data.get("category"),
                tags=data.get("tags"),
                created_by=data.get("created_by")
            )

        except Exception as e:
            logger.error(f"Failed to import schema from {file_path}: {e}")
            return False

    def get_schema_stats(self) -> dict[str, Any]:
        """Get statistics about registered schemas.
        
        Returns:
            Dictionary with schema statistics
        """
        return {
            "total_schemas": len(self._schemas),
            "categories": len(self._schema_categories),
            "schemas_by_category": {
                cat: len(schemas) for cat, schemas in self._schema_categories.items()
            },
            "schema_versions": {
                name: info.version for name, info in self._schemas.items()
            }
        }

    def _load_builtin_schemas(self):
        """Load built-in schema definitions."""
        # Load common base schemas
        self._load_base_schemas()

        # Load tool-specific schemas
        self._load_tool_schemas()

    def _load_base_schemas(self):
        """Load base schema patterns."""
        # Common properties schema
        common_properties = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search or operation query",
                    "minLength": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "vault": {
                    "type": "string",
                    "description": "Optional vault name to operate on"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Response format"
                }
            }
        }

        self.register_schema(
            name="common_properties",
            schema=common_properties,
            description="Common properties used across multiple tools",
            category="base",
            tags=["common", "base"],
            created_by="jarvis-builtin"
        )

    def _load_tool_schemas(self):
        """Load tool-specific schema patterns."""
        # Search tool schema
        search_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "minLength": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                },
                "vault": {
                    "type": "string",
                    "description": "Optional vault name to search within"
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Minimum similarity score (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Response format"
                }
            },
            "required": ["query"]
        }

        self.register_schema(
            name="search_tool",
            schema=search_schema,
            description="Schema for search-based tools",
            category="search",
            tags=["search", "query"],
            created_by="jarvis-builtin"
        )

        # Analytics tool schema
        analytics_schema = {
            "type": "object",
            "properties": {
                "vault": {
                    "type": "string",
                    "description": "Vault name to analyze",
                    "default": "default"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Response format"
                },
                "cache": {
                    "type": "boolean",
                    "description": "Whether to use cached results",
                    "default": True
                }
            }
        }

        self.register_schema(
            name="analytics_tool",
            schema=analytics_schema,
            description="Schema for analytics tools",
            category="analytics",
            tags=["analytics", "analysis"],
            created_by="jarvis-builtin"
        )

        # Utility tool schema
        utility_schema = {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["json", "markdown"],
                    "default": "markdown",
                    "description": "Response format"
                }
            }
        }

        self.register_schema(
            name="utility_tool",
            schema=utility_schema,
            description="Schema for utility tools",
            category="utility",
            tags=["utility", "helper"],
            created_by="jarvis-builtin"
        )


# Global schema manager instance
_global_manager: SchemaManager | None = None


def get_schema_manager(config: dict[str, Any] | None = None) -> SchemaManager:
    """Get the global schema manager instance.
    
    Args:
        config: Optional configuration for first-time initialization
        
    Returns:
        Global schema manager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = SchemaManager(config)
    return _global_manager


def reset_schema_manager() -> None:
    """Reset the global schema manager (mainly for testing)."""
    global _global_manager
    _global_manager = None
