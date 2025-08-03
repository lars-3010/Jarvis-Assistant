"""
Custom exception classes for Jarvis Assistant.
"""

class JarvisError(Exception):
    """Base exception for all Jarvis Assistant errors."""
    pass


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
