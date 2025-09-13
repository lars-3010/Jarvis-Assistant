"""
Extension-specific error classes for Jarvis Assistant.

This module defines custom exceptions for extension system operations.
"""



class ExtensionError(Exception):
    """Base exception for extension-related errors."""

    def __init__(self, message: str, extension_name: str | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.extension_name = extension_name
        self.cause = cause


class ExtensionLoadError(ExtensionError):
    """Exception raised when an extension fails to load."""
    pass


class ExtensionInitializationError(ExtensionError):
    """Exception raised when an extension fails to initialize."""
    pass


class ExtensionConfigurationError(ExtensionError):
    """Exception raised when an extension has invalid configuration."""
    pass


class ExtensionDependencyError(ExtensionError):
    """Exception raised when an extension's dependencies are not met."""
    pass


class ExtensionToolError(ExtensionError):
    """Exception raised when an extension tool fails to execute."""

    def __init__(self, message: str, tool_name: str, extension_name: str | None = None, cause: Exception | None = None):
        super().__init__(message, extension_name, cause)
        self.tool_name = tool_name


class ExtensionNotFoundError(ExtensionError):
    """Exception raised when a requested extension is not found."""
    pass


class ExtensionAlreadyLoadedError(ExtensionError):
    """Exception raised when attempting to load an already loaded extension."""
    pass
