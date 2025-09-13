"""
Schema Registry for MCP Tools.

This module provides a registry for managing and retrieving schemas
across the MCP tool ecosystem.
"""

import json
from pathlib import Path
from typing import Any

from jarvis.utils.logging import setup_logging

from .manager import SchemaInfo, SchemaManager
from .validator import SchemaValidator, ValidationResult

logger = setup_logging(__name__)


class SchemaRegistry:
    """Registry for managing tool schemas."""

    def __init__(self, schema_manager: SchemaManager | None = None):
        """Initialize schema registry.
        
        Args:
            schema_manager: Optional schema manager instance
        """
        self.manager = schema_manager or SchemaManager()
        self.validator = SchemaValidator()
        self._tool_schema_mapping: dict[str, str] = {}

        logger.info("Schema registry initialized")

    def register_tool_schema(
        self,
        tool_name: str,
        schema: dict[str, Any],
        schema_name: str | None = None,
        version: str = "1.0.0",
        description: str | None = None,
        category: str | None = None
    ) -> bool:
        """Register a schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            schema: JSON schema definition
            schema_name: Optional custom schema name (defaults to tool_name + "_schema")
            version: Schema version
            description: Optional description
            category: Optional category
            
        Returns:
            True if registration successful
        """
        if schema_name is None:
            schema_name = f"{tool_name}_schema"

        # Validate the schema first
        validation_result = self.validator.validate_tool_schema(tool_name, schema)
        if not validation_result.is_valid:
            logger.error(f"Schema validation failed for tool {tool_name}:")
            for error in validation_result.errors:
                logger.error(f"  {error.path}: {error.message}")
            return False

        # Log warnings if any
        if validation_result.has_warnings:
            logger.warning(f"Schema validation warnings for tool {tool_name}:")
            for warning in validation_result.warnings:
                logger.warning(f"  {warning.path}: {warning.message}")

        # Register with schema manager
        success = self.manager.register_schema(
            name=schema_name,
            schema=schema,
            version=version,
            description=description or f"Schema for {tool_name}",
            category=category or self._infer_category(tool_name),
            tags=[tool_name, "tool"],
            created_by="jarvis-tool-registry"
        )

        if success:
            # Map tool to schema
            self._tool_schema_mapping[tool_name] = schema_name
            logger.info(f"Registered schema for tool: {tool_name} -> {schema_name}")

        return success

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get schema for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Schema definition or None if not found
        """
        schema_name = self._tool_schema_mapping.get(tool_name)
        if schema_name:
            return self.manager.get_schema_definition(schema_name)
        return None

    def validate_tool_input(self, tool_name: str, input_data: Any) -> ValidationResult:
        """Validate input data for a tool.
        
        Args:
            tool_name: Name of the tool
            input_data: Input data to validate
            
        Returns:
            Validation result
        """
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return ValidationResult(
                is_valid=False,
                errors=[],
                warnings=[],
                info=[]
            )

        return self.validator.validate_data(input_data, schema)

    def list_tool_schemas(self) -> list[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names with registered schemas
        """
        return list(self._tool_schema_mapping.keys())

    def unregister_tool_schema(self, tool_name: str) -> bool:
        """Unregister a tool's schema.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if unregistration successful
        """
        schema_name = self._tool_schema_mapping.get(tool_name)
        if not schema_name:
            logger.warning(f"No schema found for tool: {tool_name}")
            return False

        # Unregister from manager
        success = self.manager.unregister_schema(schema_name)

        if success:
            # Remove mapping
            del self._tool_schema_mapping[tool_name]
            logger.info(f"Unregistered schema for tool: {tool_name}")

        return success

    def get_schema_info(self, tool_name: str) -> SchemaInfo | None:
        """Get detailed schema information for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Schema information or None if not found
        """
        schema_name = self._tool_schema_mapping.get(tool_name)
        if schema_name:
            return self.manager.get_schema(schema_name)
        return None

    def search_tool_schemas(
        self,
        query: str | None = None,
        category: str | None = None
    ) -> list[str]:
        """Search for tools by schema criteria.
        
        Args:
            query: Optional text query
            category: Optional category filter
            
        Returns:
            List of matching tool names
        """
        matching_tools = []

        for tool_name, schema_name in self._tool_schema_mapping.items():
            schema_info = self.manager.get_schema(schema_name)
            if not schema_info:
                continue

            # Category filter
            if category and schema_info.category != category:
                continue

            # Query filter
            if query:
                query_lower = query.lower()
                searchable_text = [
                    tool_name,
                    schema_info.description or "",
                    " ".join(schema_info.tags or [])
                ]

                if not any(query_lower in text.lower() for text in searchable_text):
                    continue

            matching_tools.append(tool_name)

        return matching_tools

    def validate_all_tool_schemas(self) -> dict[str, ValidationResult]:
        """Validate all registered tool schemas.
        
        Returns:
            Dictionary mapping tool names to validation results
        """
        results = {}

        for tool_name in self.list_tool_schemas():
            schema = self.get_tool_schema(tool_name)
            if schema:
                results[tool_name] = self.validator.validate_tool_schema(tool_name, schema)

        return results

    def export_tool_schemas(self, output_dir: Path) -> dict[str, bool]:
        """Export all tool schemas to files.
        
        Args:
            output_dir: Directory to save schema files
            
        Returns:
            Dictionary mapping tool names to export success status
        """
        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        for tool_name in self.list_tool_schemas():
            try:
                schema_info = self.get_schema_info(tool_name)
                if schema_info:
                    output_file = output_dir / f"{tool_name}_schema.json"
                    export_result = self.manager.export_schema(schema_info.name, output_file)
                    results[tool_name] = export_result is not None
                else:
                    results[tool_name] = False
            except Exception as e:
                logger.error(f"Failed to export schema for {tool_name}: {e}")
                results[tool_name] = False

        return results

    def import_tool_schemas(self, input_dir: Path) -> dict[str, bool]:
        """Import tool schemas from files.
        
        Args:
            input_dir: Directory containing schema files
            
        Returns:
            Dictionary mapping tool names to import success status
        """
        results = {}

        if not input_dir.exists():
            logger.error(f"Import directory does not exist: {input_dir}")
            return results

        schema_files = list(input_dir.glob("*_schema.json"))

        for schema_file in schema_files:
            try:
                # Extract tool name from filename
                tool_name = schema_file.stem.replace("_schema", "")

                # Import schema
                success = self.manager.import_schema(schema_file)
                if success:
                    # Try to map imported schema to tool
                    schema_data = json.loads(schema_file.read_text())
                    schema_name = schema_data.get("name")
                    if schema_name:
                        self._tool_schema_mapping[tool_name] = schema_name

                results[tool_name] = success

            except Exception as e:
                logger.error(f"Failed to import schema from {schema_file}: {e}")
                results[schema_file.stem] = False

        return results

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the schema registry.
        
        Returns:
            Dictionary with registry statistics
        """
        base_stats = self.manager.get_schema_stats()

        # Add registry-specific stats
        tool_stats = {
            "total_tools": len(self._tool_schema_mapping),
            "tools_by_category": {},
            "validation_summary": {}
        }

        # Count tools by category
        for tool_name in self._tool_schema_mapping:
            schema_info = self.get_schema_info(tool_name)
            if schema_info and schema_info.category:
                category = schema_info.category
                if category not in tool_stats["tools_by_category"]:
                    tool_stats["tools_by_category"][category] = 0
                tool_stats["tools_by_category"][category] += 1

        # Validation summary
        validation_results = self.validate_all_tool_schemas()
        valid_count = sum(1 for r in validation_results.values() if r.is_valid)
        warning_count = sum(1 for r in validation_results.values() if r.has_warnings)
        error_count = sum(1 for r in validation_results.values() if r.has_errors)

        tool_stats["validation_summary"] = {
            "valid_schemas": valid_count,
            "schemas_with_warnings": warning_count,
            "schemas_with_errors": error_count,
            "validation_rate": valid_count / len(validation_results) * 100 if validation_results else 0
        }

        return {**base_stats, **tool_stats}

    def _infer_category(self, tool_name: str) -> str:
        """Infer category from tool name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Inferred category
        """
        tool_lower = tool_name.lower()

        if "search" in tool_lower:
            return "search"
        elif "analytic" in tool_lower:
            return "analytics"
        elif "graph" in tool_lower:
            return "graph"
        elif any(word in tool_lower for word in ["read", "list", "vault"]):
            return "vault"
        elif any(word in tool_lower for word in ["health", "metric", "performance"]):
            return "utility"
        else:
            return "general"


# Global registry instance
_global_registry: SchemaRegistry | None = None


def get_schema_registry(schema_manager: SchemaManager | None = None) -> SchemaRegistry:
    """Get the global schema registry instance.
    
    Args:
        schema_manager: Optional schema manager for first-time initialization
        
    Returns:
        Global schema registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SchemaRegistry(schema_manager)
    return _global_registry


def reset_schema_registry() -> None:
    """Reset the global schema registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
