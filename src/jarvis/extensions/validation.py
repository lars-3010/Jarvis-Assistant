"""
Extension configuration validation for Jarvis Assistant.

This module provides validation functionality for extension configurations
and system requirements.
"""

from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError

from jarvis.extensions.interfaces import ExtensionMetadata
from jarvis.utils.config import JarvisSettings, ValidationResult
import logging

logger = logging.getLogger(__name__)


class ExtensionValidator:
    """Validator for extension configurations and requirements."""

    def __init__(self, settings: JarvisSettings):
        """Initialize the extension validator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings

    def validate_extension_system(self) -> ValidationResult:
        """Validate the extension system configuration.
        
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        if not self.settings.extensions_enabled:
            result.warnings.append("Extensions are disabled")
            return result

        # Validate extensions directory
        extensions_dir = self.settings.get_extensions_directory()
        if not extensions_dir.exists():
            result.errors.append(f"Extensions directory does not exist: {extensions_dir}")
            result.valid = False
        elif not extensions_dir.is_dir():
            result.errors.append(f"Extensions path is not a directory: {extensions_dir}")
            result.valid = False

        # Validate auto-load extensions
        if self.settings.extensions_auto_load:
            missing_extensions = self._check_auto_load_extensions(extensions_dir)
            if missing_extensions:
                result.warnings.extend([
                    f"Auto-load extension not found: {ext}" for ext in missing_extensions
                ])

        # Validate AI extension settings if enabled
        if self.settings.ai_extension_enabled:
            ai_validation = self._validate_ai_extension_config()
            result.errors.extend(ai_validation.errors)
            result.warnings.extend(ai_validation.warnings)
            if not ai_validation.valid:
                result.valid = False

        # Validate extension configurations
        config_validation = self._validate_extension_configs()
        result.errors.extend(config_validation.errors)
        result.warnings.extend(config_validation.warnings)
        if not config_validation.valid:
            result.valid = False

        return result

    def validate_extension_metadata(self, metadata: ExtensionMetadata) -> ValidationResult:
        """Validate extension metadata.
        
        Args:
            metadata: Extension metadata to validate
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Required fields
        if not metadata.name:
            result.errors.append("Extension name is required")
            result.valid = False

        if not metadata.version:
            result.errors.append("Extension version is required")
            result.valid = False

        # Name validation
        if metadata.name and not self._is_valid_extension_name(metadata.name):
            result.errors.append(f"Invalid extension name: {metadata.name}")
            result.valid = False

        # Version validation
        if metadata.version and not self._is_valid_version(metadata.version):
            result.warnings.append(f"Extension version format may be invalid: {metadata.version}")

        # Dependencies validation
        for dep in metadata.dependencies:
            if not self._is_valid_extension_name(dep):
                result.errors.append(f"Invalid dependency name: {dep}")
                result.valid = False

        # Configuration schema validation
        if metadata.configuration_schema:
            try:
                jsonschema.Draft7Validator.check_schema(metadata.configuration_schema)
            except JsonSchemaValidationError as e:
                result.errors.append(f"Invalid configuration schema: {e.message}")
                result.valid = False

        return result

    def validate_extension_config(self, extension_name: str, config: dict[str, Any],
                                 schema: dict[str, Any] | None = None) -> ValidationResult:
        """Validate configuration for a specific extension.
        
        Args:
            extension_name: Name of the extension
            config: Configuration to validate
            schema: Optional JSON schema for validation
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Schema validation if provided
        if schema:
            try:
                jsonschema.validate(config, schema)
            except JsonSchemaValidationError as e:
                result.errors.append(f"Configuration validation failed: {e.message}")
                result.valid = False

        # Extension-specific validation
        if extension_name == "ai":
            ai_result = self._validate_ai_extension_specific_config(config)
            result.errors.extend(ai_result.errors)
            result.warnings.extend(ai_result.warnings)
            if not ai_result.valid:
                result.valid = False

        return result

    def check_system_requirements(self, metadata: ExtensionMetadata) -> ValidationResult:
        """Check if system meets extension requirements.
        
        Args:
            metadata: Extension metadata with requirements
            
        Returns:
            ValidationResult with requirement check status
        """
        result = ValidationResult()

        # Check required services
        for service in metadata.required_services:
            if not self._is_service_available(service):
                result.errors.append(f"Required service not available: {service}")
                result.valid = False

        # Check optional services
        for service in metadata.optional_services:
            if not self._is_service_available(service):
                result.warnings.append(f"Optional service not available: {service}")

        # AI extension specific requirements
        if metadata.name == "ai" and self.settings.ai_extension_enabled:
            ai_requirements = self._check_ai_requirements()
            result.errors.extend(ai_requirements.errors)
            result.warnings.extend(ai_requirements.warnings)
            if not ai_requirements.valid:
                result.valid = False

        return result

    def _check_auto_load_extensions(self, extensions_dir: Path) -> list[str]:
        """Check which auto-load extensions are missing.
        
        Args:
            extensions_dir: Path to extensions directory
            
        Returns:
            List of missing extension names
        """
        missing = []

        for ext_name in self.settings.extensions_auto_load:
            ext_path = extensions_dir / ext_name
            if not ext_path.exists() or not ext_path.is_dir():
                missing.append(ext_name)
            else:
                # Check for main.py or __init__.py
                main_file = ext_path / "main.py"
                init_file = ext_path / "__init__.py"
                if not main_file.exists() and not init_file.exists():
                    missing.append(ext_name)

        return missing

    def _validate_ai_extension_config(self) -> ValidationResult:
        """Validate AI extension configuration.
        
        Returns:
            ValidationResult with AI extension validation status
        """
        result = ValidationResult()

        # Memory limits
        if self.settings.ai_max_memory_gb <= 0:
            result.errors.append("AI max memory must be positive")
            result.valid = False
        elif self.settings.ai_max_memory_gb < 4:
            result.warnings.append("AI max memory is quite low (< 4GB), may affect performance")

        # Timeout validation
        if self.settings.ai_timeout_seconds <= 0:
            result.errors.append("AI timeout must be positive")
            result.valid = False
        elif self.settings.ai_timeout_seconds < 10:
            result.warnings.append("AI timeout is quite short (< 10s), may cause timeouts")

        # LLM provider validation
        valid_providers = ["ollama", "huggingface"]
        if self.settings.ai_llm_provider not in valid_providers:
            result.errors.append(f"Invalid LLM provider: {self.settings.ai_llm_provider}")
            result.valid = False

        # Model validation
        if not self.settings.ai_llm_models:
            result.errors.append("At least one LLM model must be specified")
            result.valid = False

        return result

    def _validate_extension_configs(self) -> ValidationResult:
        """Validate all extension-specific configurations.
        
        Returns:
            ValidationResult with configuration validation status
        """
        result = ValidationResult()

        for ext_name, config in self.settings.extensions_config.items():
            if not isinstance(config, dict):
                result.errors.append(f"Extension config for {ext_name} must be a dictionary")
                result.valid = False
                continue

            # Validate structure
            if not self._is_valid_extension_name(ext_name):
                result.warnings.append(f"Extension name in config may be invalid: {ext_name}")

        return result

    def _validate_ai_extension_specific_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate AI extension specific configuration.
        
        Args:
            config: AI extension configuration
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult()

        # Validate ollama settings if present
        if "ollama" in config:
            ollama_config = config["ollama"]
            if "url" in ollama_config and not self._is_valid_url(ollama_config["url"]):
                result.errors.append("Invalid Ollama URL")
                result.valid = False

        # Validate model settings
        if "models" in config:
            models = config["models"]
            if not isinstance(models, list):
                result.errors.append("Models configuration must be a list")
                result.valid = False
            elif not models:
                result.warnings.append("No models configured")

        return result

    def _check_ai_requirements(self) -> ValidationResult:
        """Check AI extension system requirements.
        
        Returns:
            ValidationResult with requirement check status
        """
        result = ValidationResult()

        # Check memory (simplified - would need psutil for real check)
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().total / (1024**3)
            if available_memory_gb < self.settings.ai_max_memory_gb:
                result.warnings.append(
                    f"System memory ({available_memory_gb:.1f}GB) is less than configured max "
                    f"({self.settings.ai_max_memory_gb}GB)"
                )
        except ImportError:
            result.warnings.append("Cannot check system memory (psutil not available)")

        # Check if Ollama is accessible (if using ollama provider)
        if self.settings.ai_llm_provider == "ollama":
            if not self._check_ollama_availability():
                result.warnings.append("Ollama server not accessible")

        return result

    def _is_valid_extension_name(self, name: str) -> bool:
        """Check if extension name is valid.
        
        Args:
            name: Extension name to validate
            
        Returns:
            True if name is valid
        """
        if not name or not isinstance(name, str):
            return False

        # Basic validation: alphanumeric, hyphens, underscores
        import re
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name))

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid.
        
        Args:
            version: Version string to validate
            
        Returns:
            True if version is valid
        """
        if not version or not isinstance(version, str):
            return False

        # Simple semver-like validation
        import re
        return bool(re.match(r'^\d+\.\d+\.\d+.*$', version))

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid
        """
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))

    def _is_service_available(self, service_name: str) -> bool:
        """Check if a service is available.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if service is available
        """
        # This is a simplified check - in practice, you would check
        # the service container or perform actual service discovery
        known_services = {
            "vector_searcher", "graph_database", "vault_reader",
            "health_checker", "metrics", "vector_encoder"
        }
        return service_name in known_services

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is available.
        
        Returns:
            True if Ollama is accessible
        """
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
