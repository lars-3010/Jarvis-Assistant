"""
Custom exception classes for Jarvis Assistant.
"""

from typing import Any


class JarvisError(Exception):
    """Base exception for all Jarvis Assistant errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        suggestions: list[str] | None = None,
        context: dict[str, Any] | None = None
    ):
        """Initialize Jarvis error with enhanced information.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            suggestions: List of suggested remediation steps
            context: Additional context information for debugging
        """
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__.upper()
        self.suggestions = suggestions or []
        self.context = context or {}


class ConfigurationError(JarvisError):
    """Raised when there is an issue with the application configuration."""
    pass


class ServiceError(JarvisError):
    """Base exception for errors occurring in service layers."""
    pass


class ServiceUnavailableError(ServiceError):
    """Raised when a required external service (e.g., Neo4j) is unavailable."""
    pass


class ValidationError(JarvisError):
    """Raised when input data fails validation."""
    pass


class ToolExecutionError(JarvisError):
    """Raised when an MCP tool encounters an error during execution."""
    pass


class PluginError(JarvisError):
    """Raised when there's a plugin-related error."""
    pass


# Database-specific errors
class DatabaseError(ServiceError):
    """Base exception for database-related errors."""
    pass


class DatabaseInitializationError(DatabaseError):
    """Raised when database initialization fails."""
    pass


class DatabasePermissionError(DatabaseError):
    """Raised when database access is denied due to permissions."""
    pass


class DatabaseCorruptionError(DatabaseError):
    """Raised when database corruption is detected."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class DiskSpaceError(DatabaseError):
    """Raised when insufficient disk space is available."""
    pass
