"""
Analytics-specific error classes.

This module defines custom exceptions for the analytics service,
providing structured error handling and recovery guidance.
"""

from typing import Optional, Dict, Any


class AnalyticsError(Exception):
    """Base exception for analytics operations."""
    
    def __init__(
        self,
        message: str,
        component: str,
        error_type: str = "unknown",
        recovery_action: str = "retry operation",
        impact_level: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.component = component
        self.error_type = error_type
        self.recovery_action = recovery_action
        self.impact_level = impact_level
        self.context = context or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "error_type": self.error_type,
            "message": self.message,
            "recovery_action": self.recovery_action,
            "impact_level": self.impact_level,
            "context": self.context
        }


class VaultNotFoundError(AnalyticsError):
    """Raised when specified vault cannot be found."""
    
    def __init__(self, vault_name: str):
        super().__init__(
            message=f"Vault '{vault_name}' not found or inaccessible",
            component="vault_reader",
            error_type="vault_not_found",
            recovery_action="verify vault name and accessibility",
            impact_level="high",
            context={"vault_name": vault_name}
        )


class AnalysisTimeoutError(AnalyticsError):
    """Raised when analysis takes too long to complete."""
    
    def __init__(self, component: str, timeout_seconds: float):
        super().__init__(
            message=f"Analysis timed out after {timeout_seconds} seconds",
            component=component,
            error_type="timeout",
            recovery_action="increase timeout or enable sampling for large vaults",
            impact_level="medium",
            context={"timeout_seconds": timeout_seconds}
        )


class InsufficientDataError(AnalyticsError):
    """Raised when vault has insufficient data for analysis."""
    
    def __init__(self, component: str, minimum_required: int, actual: int):
        super().__init__(
            message=f"Insufficient data: need at least {minimum_required} items, got {actual}",
            component=component,
            error_type="insufficient_data",
            recovery_action="ensure vault has adequate content for analysis",
            impact_level="low",
            context={"minimum_required": minimum_required, "actual": actual}
        )


class CacheError(AnalyticsError):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, cache_key: str, cause: Optional[str] = None):
        message = f"Cache {operation} failed for key '{cache_key}'"
        if cause:
            message += f": {cause}"
            
        super().__init__(
            message=message,
            component="analytics_cache",
            error_type="cache_error",
            recovery_action="clear cache and retry",
            impact_level="low",
            context={"operation": operation, "cache_key": cache_key, "cause": cause}
        )


class ServiceUnavailableError(AnalyticsError):
    """Raised when required service is unavailable."""
    
    def __init__(self, service_name: str, operation: str):
        super().__init__(
            message=f"Service '{service_name}' unavailable for {operation}",
            component="service_integration",
            error_type="service_unavailable",
            recovery_action="check service health and restart if needed",
            impact_level="high",
            context={"service_name": service_name, "operation": operation}
        )


class ConfigurationError(AnalyticsError):
    """Raised when analytics configuration is invalid."""
    
    def __init__(self, config_key: str, issue: str):
        super().__init__(
            message=f"Configuration error for '{config_key}': {issue}",
            component="analytics_config",
            error_type="configuration_error",
            recovery_action="check and correct configuration settings",
            impact_level="high",
            context={"config_key": config_key, "issue": issue}
        )


class ModelError(AnalyticsError):
    """Raised when ML model operations fail."""
    
    def __init__(self, model_name: str, operation: str, cause: Optional[str] = None):
        message = f"Model '{model_name}' failed during {operation}"
        if cause:
            message += f": {cause}"
            
        super().__init__(
            message=message,
            component="ml_models",
            error_type="model_error",
            recovery_action="check model availability and resources",
            impact_level="medium",
            context={"model_name": model_name, "operation": operation, "cause": cause}
        )


class DataCorruptionError(AnalyticsError):
    """Raised when data appears corrupted or invalid."""
    
    def __init__(self, data_type: str, issue: str):
        super().__init__(
            message=f"Data corruption detected in {data_type}: {issue}",
            component="data_validation",
            error_type="data_corruption",
            recovery_action="regenerate corrupted data or restore from backup",
            impact_level="high",
            context={"data_type": data_type, "issue": issue}
        )


class ResourceExhaustionError(AnalyticsError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource_type: str, current_usage: str, limit: str):
        super().__init__(
            message=f"Resource exhaustion: {resource_type} usage {current_usage} exceeds limit {limit}",
            component="resource_management",
            error_type="resource_exhaustion",
            recovery_action="reduce analysis scope or increase resource limits",
            impact_level="high",
            context={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit
            }
        )


# Error code mapping for structured error handling
ERROR_CODES = {
    "vault_not_found": VaultNotFoundError,
    "timeout": AnalysisTimeoutError,
    "insufficient_data": InsufficientDataError,
    "cache_error": CacheError,
    "service_unavailable": ServiceUnavailableError,
    "configuration_error": ConfigurationError,
    "model_error": ModelError,
    "data_corruption": DataCorruptionError,
    "resource_exhaustion": ResourceExhaustionError,
}