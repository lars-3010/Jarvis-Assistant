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


class AreasNotFoundError(VaultValidationError):
    """Areas/ folder not found in vault."""
    
    def __init__(self, vault_path: str, areas_folder_name: str = "Areas"):
        """
        Initialize AreasNotFoundError with actionable guidance.
        
        Args:
            vault_path: Path to the vault where Areas/ folder was not found
            areas_folder_name: Name of the expected Areas folder (default: "Areas")
        """
        message = (
            f"🔍 Areas/ folder not found in vault: {vault_path}\n\n"
            f"📁 REQUIRED FOLDER STRUCTURE:\n"
            f"Your vault needs an '{areas_folder_name}/' folder containing structured knowledge content.\n\n"
            f"🛠️  QUICK SETUP GUIDE:\n"
            f"1. Create folder: {vault_path}/{areas_folder_name}/\n"
            f"2. Add knowledge subdirectories (e.g., Computer Science/, Business/)\n"
            f"3. Move your best knowledge notes into these subdirectories\n"
            f"4. Keep personal content (Journal/, People/) outside Areas/ for privacy\n\n"
            f"💡 This ensures only structured knowledge is used for dataset generation,\n"
            f"   protecting your personal journals and private notes."
        )
        super().__init__(message, vault_path, "areas_not_found")
        self.areas_folder_name = areas_folder_name
        self.expected_path = f"{vault_path}/{areas_folder_name}"


class InsufficientAreasContentError(InsufficientDataError):
    """Insufficient content in Areas/ folder for dataset generation."""
    
    def __init__(self, areas_folder_path: str, areas_count: int, required_minimum: int = 5):
        """
        Initialize InsufficientAreasContentError with actionable guidance.
        
        Args:
            areas_folder_path: Path to the Areas/ folder that has insufficient content
            areas_count: Actual number of markdown files found in Areas/
            required_minimum: Minimum number of files required for dataset generation
        """
        message = (
            f"📊 Insufficient content in Areas/ folder: {areas_count} < {required_minimum} notes\n\n"
            f"📁 CURRENT STATUS:\n"
            f"• Areas/ folder found: {areas_folder_path}\n"
            f"• Markdown files found: {areas_count}\n"
            f"• Minimum required: {required_minimum}\n\n"
            f"🛠️  HOW TO ADD MORE CONTENT:\n"
            f"1. Move knowledge notes from other folders:\n"
            f"   • Journal/ → Areas/ (for knowledge entries)\n"
            f"   • Inbox/ → Areas/ (for processed knowledge)\n"
            f"   • Root folder → Areas/ (for unorganized knowledge)\n"
            f"2. Create new knowledge notes in Areas/ subdirectories\n"
            f"3. Ensure .md files contain substantial content (not empty)\n"
            f"4. Organize content into knowledge domains (Computer Science/, Business/, etc.)\n\n"
            f"💡 Focus on moving your best knowledge content to Areas/ first.\n"
            f"   Personal journals and private notes should stay outside Areas/ for privacy."
        )
        super().__init__(message, required_minimum, areas_count)
        self.areas_folder_path = areas_folder_path
        self.areas_count = areas_count
