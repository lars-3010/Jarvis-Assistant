"""
Custom exception classes for dataset generation.

This module defines the exception hierarchy for dataset generation operations,
providing specific error types for different failure scenarios.
"""

from jarvis.utils.errors import JarvisError


class DatasetGenerationError(JarvisError):
    """Base exception for all dataset generation errors."""
    pass


class LinkExtractionError(DatasetGenerationError):
    """Errors during link extraction process."""
    pass


class BrokenLinkError(LinkExtractionError):
    """Link points to non-existent target."""

    def __init__(self, message: str, link_source: str = None, link_target: str = None):
        super().__init__(message)
        self.link_source = link_source
        self.link_target = link_target


class CircularLinkError(LinkExtractionError):
    """Circular reference detected in links."""

    def __init__(self, message: str, cycle_path: list = None):
        super().__init__(message)
        self.cycle_path = cycle_path or []


class FeatureEngineeringError(DatasetGenerationError):
    """Errors during feature computation."""
    pass


class EmbeddingError(FeatureEngineeringError):
    """Error computing semantic embeddings."""

    def __init__(self, message: str, text_snippet: str = None, model_name: str = None):
        super().__init__(message)
        self.text_snippet = text_snippet
        self.model_name = model_name


class GraphMetricError(FeatureEngineeringError):
    """Error computing graph-based metrics."""

    def __init__(self, message: str, metric_name: str = None, node_id: str = None):
        super().__init__(message)
        self.metric_name = metric_name
        self.node_id = node_id


class DataQualityError(DatasetGenerationError):
    """Data quality validation failures."""
    pass


class InsufficientDataError(DataQualityError):
    """Not enough data for meaningful dataset generation."""

    def __init__(self, message: str, required_minimum: int = None, actual_count: int = None):
        super().__init__(message)
        self.required_minimum = required_minimum
        self.actual_count = actual_count


class CorruptedDataError(DataQualityError):
    """Data corruption detected during processing."""

    def __init__(self, message: str, file_path: str = None, corruption_type: str = None):
        super().__init__(message)
        self.file_path = file_path
        self.corruption_type = corruption_type


class VaultValidationError(DatasetGenerationError):
    """Vault structure or accessibility validation error."""

    def __init__(self, message: str, vault_path: str = None, validation_type: str = None):
        super().__init__(message)
        self.vault_path = vault_path
        self.validation_type = validation_type


class SamplingError(DatasetGenerationError):
    """Error during negative sampling or dataset balancing."""

    def __init__(self, message: str, sampling_strategy: str = None, target_ratio: float = None):
        super().__init__(message)
        self.sampling_strategy = sampling_strategy
        self.target_ratio = target_ratio


class MemoryError(DatasetGenerationError):
    """Memory-related error during processing."""

    def __init__(self, message: str, memory_usage_mb: float = None, memory_limit_mb: float = None):
        super().__init__(message)
        self.memory_usage_mb = memory_usage_mb
        self.memory_limit_mb = memory_limit_mb


class ConfigurationError(DatasetGenerationError):
    """Configuration-related error for dataset generation."""

    def __init__(self, message: str, config_key: str = None, config_value: str = None):
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value


class TimeoutError(DatasetGenerationError):
    """Operation timeout during dataset generation."""

    def __init__(self, message: str, operation_name: str = None, timeout_seconds: float = None):
        super().__init__(message)
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
