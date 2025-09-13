"""
Schema Validation System for MCP Tools.

This module provides comprehensive validation for MCP tool schemas and data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Represents a validation error or warning."""

    level: ValidationLevel
    message: str
    path: str
    value: Any = None
    expected: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "path": self.path,
            "value": self.value,
            "expected": self.expected
        }


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]
    info: list[ValidationError]

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.errors) + len(self.warnings) + len(self.info)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "total_issues": self.total_issues,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info]
        }


class SchemaValidator:
    """Advanced schema validator with custom rules."""

    def __init__(self, strict: bool = True, enable_warnings: bool = True):
        """Initialize validator.
        
        Args:
            strict: Whether to use strict validation
            enable_warnings: Whether to generate warnings
        """
        self.strict = strict
        self.enable_warnings = enable_warnings
        self._custom_validators: dict[str, callable] = {}

        # Register built-in custom validators
        self._register_builtin_validators()

    def validate_schema(self, schema: dict[str, Any]) -> ValidationResult:
        """Validate a JSON schema definition.
        
        Args:
            schema: Schema to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        info = []

        try:
            # Basic JSON Schema validation
            Draft7Validator.check_schema(schema)

            # Custom schema validation
            custom_issues = self._validate_schema_custom(schema)
            errors.extend([e for e in custom_issues if e.level == ValidationLevel.ERROR])
            warnings.extend([e for e in custom_issues if e.level == ValidationLevel.WARNING])
            info.extend([e for e in custom_issues if e.level == ValidationLevel.INFO])

        except jsonschema.SchemaError as e:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"JSON Schema validation failed: {e.message}",
                path=".".join(str(p) for p in e.path) if e.path else "root",
                value=getattr(e, 'instance', None)
            ))
        except Exception as e:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Schema validation error: {e}",
                path="root"
            ))

        is_valid = len(errors) == 0
        if self.strict and len(warnings) > 0:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info
        )

    def validate_data(self, data: Any, schema: dict[str, Any]) -> ValidationResult:
        """Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema to validate against
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        info = []

        try:
            # Create validator
            validator = Draft7Validator(schema)

            # Validate data
            for error in validator.iter_errors(data):
                error_path = ".".join(str(p) for p in error.path) if error.path else "root"

                # Determine if this is an error or warning
                level = self._classify_validation_issue(error)

                validation_error = ValidationError(
                    level=level,
                    message=error.message,
                    path=error_path,
                    value=error.instance,
                    expected=getattr(error.schema, 'get', lambda x: None)("type")
                )

                if level == ValidationLevel.ERROR:
                    errors.append(validation_error)
                elif level == ValidationLevel.WARNING and self.enable_warnings:
                    warnings.append(validation_error)
                else:
                    info.append(validation_error)

            # Custom data validation
            custom_issues = self._validate_data_custom(data, schema)
            errors.extend([e for e in custom_issues if e.level == ValidationLevel.ERROR])
            warnings.extend([e for e in custom_issues if e.level == ValidationLevel.WARNING])
            info.extend([e for e in custom_issues if e.level == ValidationLevel.INFO])

        except Exception as e:
            errors.append(ValidationError(
                level=ValidationLevel.ERROR,
                message=f"Data validation error: {e}",
                path="root"
            ))

        is_valid = len(errors) == 0
        if self.strict and len(warnings) > 0:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info
        )

    def validate_tool_schema(self, tool_name: str, schema: dict[str, Any]) -> ValidationResult:
        """Validate a tool's input schema.
        
        Args:
            tool_name: Name of the tool
            schema: Input schema to validate
            
        Returns:
            Validation result
        """
        result = self.validate_schema(schema)

        # Add tool-specific validation
        tool_issues = self._validate_tool_specific(tool_name, schema)
        result.errors.extend([e for e in tool_issues if e.level == ValidationLevel.ERROR])
        result.warnings.extend([e for e in tool_issues if e.level == ValidationLevel.WARNING])
        result.info.extend([e for e in tool_issues if e.level == ValidationLevel.INFO])

        # Update validity
        result.is_valid = len(result.errors) == 0
        if self.strict and len(result.warnings) > 0:
            result.is_valid = False

        return result

    def add_custom_validator(self, name: str, validator_func: callable):
        """Add a custom validator function.
        
        Args:
            name: Name of the validator
            validator_func: Function that takes (data, schema) and returns List[ValidationError]
        """
        self._custom_validators[name] = validator_func
        logger.info(f"Added custom validator: {name}")

    def remove_custom_validator(self, name: str):
        """Remove a custom validator.
        
        Args:
            name: Name of the validator to remove
        """
        if name in self._custom_validators:
            del self._custom_validators[name]
            logger.info(f"Removed custom validator: {name}")

    def _classify_validation_issue(self, error: jsonschema.ValidationError) -> ValidationLevel:
        """Classify a validation issue as error, warning, or info.
        
        Args:
            error: Validation error from jsonschema
            
        Returns:
            Classification level
        """
        # Most validation failures are errors
        error_keywords = ["required", "type", "enum", "format"]
        warning_keywords = ["minimum", "maximum", "minLength", "maxLength"]

        validator = error.validator

        if validator in error_keywords:
            return ValidationLevel.ERROR
        elif validator in warning_keywords:
            return ValidationLevel.WARNING
        else:
            return ValidationLevel.ERROR  # Default to error

    def _validate_schema_custom(self, schema: dict[str, Any]) -> list[ValidationError]:
        """Apply custom schema validation rules.
        
        Args:
            schema: Schema to validate
            
        Returns:
            List of validation issues
        """
        issues = []

        # Check for required properties patterns
        if "properties" in schema:
            properties = schema["properties"]
            required = schema.get("required", [])

            # Check for common MCP patterns
            if "query" in properties and "query" not in required:
                issues.append(ValidationError(
                    level=ValidationLevel.WARNING,
                    message="Query parameter should typically be required for search tools",
                    path="properties.query"
                ))

            # Check for format consistency
            if "format" in properties:
                format_prop = properties["format"]
                if format_prop.get("type") == "string":
                    enum_values = format_prop.get("enum")
                    if enum_values and "markdown" not in enum_values:
                        issues.append(ValidationError(
                            level=ValidationLevel.INFO,
                            message="Consider including 'markdown' as a format option",
                            path="properties.format.enum"
                        ))

            # Check for description completeness
            for prop_name, prop_def in properties.items():
                if not prop_def.get("description"):
                    issues.append(ValidationError(
                        level=ValidationLevel.WARNING,
                        message=f"Property '{prop_name}' missing description",
                        path=f"properties.{prop_name}.description"
                    ))

        return issues

    def _validate_data_custom(self, data: Any, schema: dict[str, Any]) -> list[ValidationError]:
        """Apply custom data validation rules.
        
        Args:
            data: Data to validate
            schema: Schema being validated against
            
        Returns:
            List of validation issues
        """
        issues = []

        # Apply custom validators
        for name, validator_func in self._custom_validators.items():
            try:
                custom_issues = validator_func(data, schema)
                if isinstance(custom_issues, list):
                    issues.extend(custom_issues)
            except Exception as e:
                issues.append(ValidationError(
                    level=ValidationLevel.ERROR,
                    message=f"Custom validator '{name}' failed: {e}",
                    path="root"
                ))

        return issues

    def _validate_tool_specific(self, tool_name: str, schema: dict[str, Any]) -> list[ValidationError]:
        """Apply tool-specific validation rules.
        
        Args:
            tool_name: Name of the tool
            schema: Schema to validate
            
        Returns:
            List of validation issues
        """
        issues = []

        # Search tool validation
        if "search" in tool_name.lower():
            if "properties" in schema:
                properties = schema["properties"]

                # Should have query parameter
                if "query" not in properties:
                    issues.append(ValidationError(
                        level=ValidationLevel.ERROR,
                        message="Search tools should have a 'query' parameter",
                        path="properties"
                    ))

                # Should have limit parameter
                if "limit" not in properties:
                    issues.append(ValidationError(
                        level=ValidationLevel.WARNING,
                        message="Search tools should typically have a 'limit' parameter",
                        path="properties"
                    ))

        # Analytics tool validation
        elif "analytic" in tool_name.lower():
            if "properties" in schema:
                properties = schema["properties"]

                # Should have vault parameter
                if "vault" not in properties:
                    issues.append(ValidationError(
                        level=ValidationLevel.INFO,
                        message="Analytics tools often have a 'vault' parameter",
                        path="properties"
                    ))

        return issues

    def _register_builtin_validators(self):
        """Register built-in custom validators."""

        def validate_limit_ranges(data: Any, schema: dict[str, Any]) -> list[ValidationError]:
            """Validate that limit parameters have reasonable ranges."""
            issues = []

            if isinstance(data, dict) and "limit" in data:
                limit_value = data["limit"]
                if isinstance(limit_value, int):
                    if limit_value > 100:
                        issues.append(ValidationError(
                            level=ValidationLevel.WARNING,
                            message="Limit value is very high, consider performance impact",
                            path="limit",
                            value=limit_value
                        ))
                    elif limit_value < 1:
                        issues.append(ValidationError(
                            level=ValidationLevel.ERROR,
                            message="Limit value must be positive",
                            path="limit",
                            value=limit_value
                        ))

            return issues

        def validate_query_content(data: Any, schema: dict[str, Any]) -> list[ValidationError]:
            """Validate query parameter content."""
            issues = []

            if isinstance(data, dict) and "query" in data:
                query_value = data["query"]
                if isinstance(query_value, str):
                    if not query_value.strip():
                        issues.append(ValidationError(
                            level=ValidationLevel.ERROR,
                            message="Query cannot be empty or whitespace only",
                            path="query",
                            value=query_value
                        ))
                    elif len(query_value) > 1000:
                        issues.append(ValidationError(
                            level=ValidationLevel.WARNING,
                            message="Query is very long, consider breaking it down",
                            path="query",
                            value=f"<{len(query_value)} characters>"
                        ))

            return issues

        # Register validators
        self.add_custom_validator("limit_ranges", validate_limit_ranges)
        self.add_custom_validator("query_content", validate_query_content)
