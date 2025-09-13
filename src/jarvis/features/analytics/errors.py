"""
Analytics-specific error classes.

This module defines custom exceptions for the analytics service,
providing structured error information and consistent handling.
"""

from dataclasses import dataclass


class AnalyticsError(Exception):
    """Base exception for analytics operations."""

    def __init__(self, message: str, component: str, operation: str):
        super().__init__(message)
        self.component = component
        self.operation = operation


class VaultNotFoundError(AnalyticsError):
    """Raised when the specified vault is not found."""

    def __init__(self, vault_name: str):
        super().__init__(f"Vault not found: {vault_name}", "vault_reader", "vault_access")
        self.vault_name = vault_name


class AnalysisTimeoutError(AnalyticsError):
    """Raised when an analysis operation exceeds time limits."""

    def __init__(self, component: str, timeout_seconds: int):
        super().__init__(
            f"Analysis timeout after {timeout_seconds}s",
            component,
            "analysis_timeout",
        )
        self.timeout_seconds = timeout_seconds


class InsufficientDataError(AnalyticsError):
    """Raised when there isn't enough data for analysis."""

    def __init__(self, component: str, required: int, found: int):
        super().__init__(
            f"Insufficient data: required {required}, found {found}",
            component,
            "insufficient_data",
        )
        self.required = required
        self.found = found


class CacheError(AnalyticsError):
    """Raised for cache-related errors."""

    def __init__(self, component: str, message: str):
        super().__init__(message, component, "cache_operation")


class ServiceUnavailableError(AnalyticsError):
    """Raised when a required service is unavailable."""

    def __init__(self, service_name: str, operation: str):
        super().__init__(
            f"Required service '{service_name}' unavailable for {operation}",
            "service_dependency",
            operation,
        )
        self.service_name = service_name


class ConfigurationError(AnalyticsError):
    """Raised when analytics configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(message, "analytics_config", "configuration")


class ModelError(AnalyticsError):
    """Raised when a model operation fails."""

    def __init__(self, component: str, operation: str, message: str):
        super().__init__(message, component, operation)


class DataCorruptionError(AnalyticsError):
    """Raised when data appears to be corrupted or invalid."""

    def __init__(self, component: str, message: str):
        super().__init__(message, component, "data_corruption")


class ResourceExhaustionError(AnalyticsError):
    """Raised when resource limits are exceeded during analysis."""

    def __init__(self, resource: str, limit: float):
        super().__init__(f"Resource exhausted: {resource} limit {limit}", "analytics_runtime", "resource_exhaustion")
        self.resource = resource
        self.limit = limit

