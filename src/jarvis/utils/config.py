"""
Configuration management for Jarvis Assistant.

This module provides centralized configuration using Pydantic settings
with support for environment variables and .env files.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ValidationResult(BaseModel):
    valid: bool = True
    errors: list[str] = []
    warnings: list[str] = []


class JarvisSettings(BaseSettings):
    """Jarvis Assistant configuration settings."""

    # Application
    app_name: str = "jarvis-assistant"
    app_version: str = "0.2.0"
    debug: bool = Field(default=False)

    def __init__(self, **kwargs):
        """Initialize settings with quiet defaults; debug logs only when enabled."""
        super().__init__(**kwargs)
        import logging, os
        logger = logging.getLogger(__name__)
        dbg = bool(getattr(self, "debug", False))
        if dbg:
            neo4j_env_password = os.environ.get("JARVIS_NEO4J_PASSWORD")
            neo4j_env_user = os.environ.get("JARVIS_NEO4J_USER")
            neo4j_env_uri = os.environ.get("JARVIS_NEO4J_URI")
            logger.debug("JarvisSettings loaded (debug on)")
            logger.debug(
                "Env presence — PASSWORD:%s USER:%s URI:%s",
                "SET" if neo4j_env_password else "NOT SET",
                "SET" if neo4j_env_user else "NOT SET",
                "SET" if neo4j_env_uri else "NOT SET",
            )
            logger.debug("neo4j_user=%s neo4j_uri=%s", self.neo4j_user, self.neo4j_uri)

    # API settings
    api_title: str = "Jarvis Assistant"
    api_description: str = "AI assistant for Obsidian with Neo4j integration"
    api_version: str = "0.1.0"
    backend_port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["*"])

    # LLM settings
    # External (non-JARVIS_) envs supported via validation_alias
    from pydantic import AliasChoices as _AliasChoices  # type: ignore
    google_api_key: str | None = Field(default=None, validation_alias=_AliasChoices("GOOGLE_API_KEY"))
    gemini_model_id: str = Field(default="gemini-1.5-flash-latest", validation_alias=_AliasChoices("GEMINI_MODEL_ID"))

    # Vault exclusion settings
    excluded_folders: list[str] = Field(
        default=["Journaling", "Atlas/People", "Atlas/work People"],
        description="Folders to exclude from indexing and querying"
    )

    # Vault settings
    vault_path: str = Field(default="", description="Path to primary Obsidian vault")
    vault_watch: bool = Field(default=True, description="Enable file system watching")

    # Vector database settings
    vector_db_backend: str = Field(default="duckdb", description="Vector database backend (supported: duckdb)")
    vector_db_path: str = Field(default="~/.jarvis/jarvis-vector.duckdb", description="Path to DuckDB vector database file")
    vector_db_read_only: bool = Field(default=False)

    # Graph database settings
    graph_db_backend: str = Field(default="neo4j", description="Graph database backend (supported: neo4j)")
    graph_enabled: bool = Field(default=True, description="Enable graph database integration")

    # Neo4j settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")


    # Embedding settings
    embedding_model_name: str = Field(default="paraphrase-MiniLM-L6-v2", description="Sentence transformer model name")
    embedding_device: str = Field(default="mps", description="PyTorch device for embeddings")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # MCP server settings
    mcp_server_name: str = Field(default="jarvis-assistant")
    mcp_server_version: str = Field(default="0.2.0")
    mcp_cache_size: int = Field(default=100, description="Maximum number of cached MCP tool call results.")
    mcp_cache_ttl: int = Field(default=300, description="Time to live for cached MCP entries in seconds.")

    # Metrics and monitoring settings
    metrics_enabled: bool = Field(default=True, description="Enable performance metrics collection")
    metrics_sampling_rate: float = Field(default=1.0, description="Sampling rate for metrics collection (0.0-1.0)")
    metrics_retention_minutes: int = Field(default=60, description="How long to retain metrics in memory (minutes)")
    metrics_detailed_logging: bool = Field(default=False, description="Enable detailed metrics logging for debugging")

    # Indexing settings
    index_batch_size: int = Field(default=32, description="Batch size for document indexing")
    index_enqueue_all: bool = Field(default=False, description="Force reindex all documents on startup")

    # Dataset generation settings (removed)

    # Search settings
    search_default_limit: int = Field(default=10, description="Default limit for search results")
    search_similarity_threshold: float = Field(default=-10.0, description="Minimum similarity threshold for search results (negative values for cosine distance)")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Logging format")
    log_file: str = Field(default="~/.jarvis/mcp_server.log", description="Log file path")

    # Database settings
    database_path: str = Field(default="~/.jarvis/jarvis.duckdb", description="Main database file path")

    # Service container settings
    use_dependency_injection: bool = Field(default=True, description="Enable dependency injection container")
    service_logging_enabled: bool = Field(default=True, description="Enable detailed service logging")
    service_health_check_interval: int = Field(default=60, description="Service health check interval in seconds")

    # Extension system settings
    extensions_enabled: bool = Field(default=False, description="Enable extension system (Phase 0)")
    extensions_auto_load: list[str] = Field(default=[], description="List of extensions to automatically load on startup")
    extensions_directory: str = Field(default="src/jarvis/extensions", description="Directory containing extensions")
    extensions_config: dict[str, Any] = Field(
        default={},
        description="Extension-specific configuration"
    )

    # AI Extension settings (Phase 1+)
    ai_extension_enabled: bool = Field(default=False, description="Enable AI extension with LLM capabilities")
    ai_llm_provider: str = Field(default="ollama", description="LLM provider: ollama, huggingface")
    ai_llm_models: list[str] = Field(default=["llama2:7b"], description="Available LLM models")
    ai_max_memory_gb: int = Field(default=8, description="Maximum memory usage for AI operations (GB)")
    ai_timeout_seconds: int = Field(default=30, description="Timeout for AI operations (seconds)")
    ai_graphrag_enabled: bool = Field(default=False, description="Enable GraphRAG capabilities")
    ai_workflows_enabled: bool = Field(default=False, description="Enable workflow orchestration")

    # Analytics settings
    analytics_enabled: bool = Field(default=True, description="Enable vault analytics engine")
    analytics_cache_enabled: bool = Field(default=True, description="Enable analytics caching")
    analytics_cache_max_size_mb: int = Field(default=100, description="Maximum analytics cache size in MB")
    analytics_cache_ttl_minutes: int = Field(default=60, description="Analytics cache time-to-live in minutes")
    analytics_quality_scoring_algorithm: str = Field(default="comprehensive", description="Quality scoring algorithm: basic, comprehensive")
    analytics_quality_connection_weight: float = Field(default=0.3, description="Weight for connection metrics in quality scoring (0.0-1.0)")
    analytics_quality_freshness_weight: float = Field(default=0.2, description="Weight for freshness metrics in quality scoring (0.0-1.0)")
    analytics_domains_clustering_threshold: float = Field(default=0.7, description="Similarity threshold for domain clustering (0.0-1.0)")
    analytics_domains_min_cluster_size: int = Field(default=3, description="Minimum notes required to form a domain cluster")
    analytics_domains_max_domains: int = Field(default=20, description="Maximum number of domains to identify")
    analytics_max_processing_time_seconds: int = Field(default=15, description="Maximum processing time for analytics operations")
    analytics_enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing for analytics")
    analytics_sample_large_vaults: bool = Field(default=True, description="Use sampling for large vaults to improve performance")
    analytics_sample_threshold: int = Field(default=5000, description="Note count threshold for enabling sampling")

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="JARVIS_",
        extra="ignore",
    )

    def get_vault_path(self) -> Path | None:
        """Get vault path as Path object."""
        if self.vault_path:
            return Path(self.vault_path).expanduser().resolve()
        return None

    def get_vector_db_path(self) -> Path:
        """Get vector database path as Path object."""
        # Ensure the parent directory exists
        path = Path(self.vector_db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_extensions_directory(self) -> Path:
        """Get extensions directory path as Path object."""
        return Path(self.extensions_directory).expanduser().resolve()

    def get_database_path(self) -> Path:
        """Get main database path as Path object."""
        path = Path(self.database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_log_file_path(self) -> Path:
        """Get log file path as Path object."""
        path = Path(self.log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # Dataset output path not applicable; dataset tooling removed

    def validate_settings(self) -> ValidationResult:
        """Validate settings and return status information."""
        status = ValidationResult()

        # Validate vault path
        vault_path = self.get_vault_path()
        if vault_path and not vault_path.exists():
            status.errors.append(f"Vault path does not exist: {vault_path}")
            status.valid = False

        # Validate vector database directory (already handled in get_vector_db_path)

        # Validate Neo4j connection if enabled
        if self.graph_enabled:
            try:
                from neo4j import GraphDatabase as Neo4jGraphDatabase
                from neo4j.exceptions import ServiceUnavailable
                with Neo4jGraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)) as driver:
                    driver.verify_connectivity()
            except ServiceUnavailable:
                status.warnings.append("Neo4j connection failed. Graph features may be unavailable.")
            except Exception as e:
                status.warnings.append(f"Could not check Neo4j health: {e}")

        # Dataset generation configuration removed

        # Validate analytics configuration
        if self.analytics_enabled:
            # Validate weights sum to reasonable values
            total_weight = (self.analytics_quality_connection_weight +
                           self.analytics_quality_freshness_weight)
            if total_weight > 1.0:
                status.errors.append(f"Analytics quality weights sum to {total_weight:.2f}, must be ≤ 1.0")
                status.valid = False

            # Validate thresholds are in valid ranges
            if not (0.0 <= self.analytics_domains_clustering_threshold <= 1.0):
                status.errors.append("Analytics clustering threshold must be between 0.0 and 1.0")
                status.valid = False

            if self.analytics_domains_min_cluster_size < 2:
                status.errors.append("Analytics minimum cluster size must be at least 2")
                status.valid = False

            if self.analytics_max_processing_time_seconds < 1:
                status.errors.append("Analytics max processing time must be at least 1 second")
                status.valid = False

            if self.analytics_cache_max_size_mb < 1:
                status.errors.append("Analytics cache max size must be at least 1 MB")
                status.valid = False

        return status

    def get_analytics_config(self) -> dict[str, Any]:
        """Get analytics configuration as a structured dictionary."""
        return {
            "enabled": self.analytics_enabled,
            "cache": {
                "enabled": self.analytics_cache_enabled,
                "max_size_mb": self.analytics_cache_max_size_mb,
                "ttl_minutes": self.analytics_cache_ttl_minutes,
            },
            "quality": {
                "scoring_algorithm": self.analytics_quality_scoring_algorithm,
                "connection_weight": self.analytics_quality_connection_weight,
                "freshness_weight": self.analytics_quality_freshness_weight,
            },
            "domains": {
                "clustering_threshold": self.analytics_domains_clustering_threshold,
                "min_cluster_size": self.analytics_domains_min_cluster_size,
                "max_domains": self.analytics_domains_max_domains,
            },
            "performance": {
                "max_processing_time_seconds": self.analytics_max_processing_time_seconds,
                "enable_parallel_processing": self.analytics_enable_parallel_processing,
                "sample_large_vaults": self.analytics_sample_large_vaults,
                "sample_threshold": self.analytics_sample_threshold,
            }
        }


# Global settings instance
settings = JarvisSettings()


def get_settings() -> JarvisSettings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> JarvisSettings:
    """Reload settings from environment and return new instance."""
    global settings
    settings = JarvisSettings()
    return settings
