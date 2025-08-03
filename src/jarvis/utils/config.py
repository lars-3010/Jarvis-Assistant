"""
Configuration management for Jarvis Assistant.

This module provides centralized configuration using Pydantic settings
with support for environment variables and .env files.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings


from jarvis.utils.errors import ConfigurationError


class ValidationResult(BaseModel):
    valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []


class JarvisSettings(BaseSettings):
    """Jarvis Assistant configuration settings."""
    
    # Application
    app_name: str = "jarvis-assistant"
    app_version: str = "0.2.0"
    debug: bool = Field(default=False, env="JARVIS_DEBUG")
    
    def __init__(self, **kwargs):
        """Initialize settings with extensive logging for debugging."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log environment variables for debugging
        import os
        neo4j_env_password = os.environ.get("JARVIS_NEO4J_PASSWORD")
        neo4j_env_user = os.environ.get("JARVIS_NEO4J_USER")
        neo4j_env_uri = os.environ.get("JARVIS_NEO4J_URI")
        
        logger.info(f"Loading JarvisSettings...")
        logger.info(f"Environment variables found - JARVIS_NEO4J_PASSWORD: {'SET' if neo4j_env_password else 'NOT SET'}")
        logger.info(f"Environment variables found - JARVIS_NEO4J_USER: {'SET' if neo4j_env_user else 'NOT SET'}")
        logger.info(f"Environment variables found - JARVIS_NEO4J_URI: {'SET' if neo4j_env_uri else 'NOT SET'}")
        
        super().__init__(**kwargs)
        
        # Log final values
        logger.info(f"Final configuration - neo4j_user: '{self.neo4j_user}'")
        logger.info(f"Final configuration - neo4j_password: {'*' * len(self.neo4j_password)} (length: {len(self.neo4j_password)})")
        logger.info(f"Final configuration - neo4j_uri: '{self.neo4j_uri}'")

    # API settings
    api_title: str = "Jarvis Assistant"
    api_description: str = "AI assistant for Obsidian with Neo4j integration"
    api_version: str = "0.1.0"
    backend_port: int = Field(default=8000, env="JARVIS_BACKEND_PORT")
    cors_origins: List[str] = Field(default=["*"], env="JARVIS_CORS_ORIGINS")

    # LLM settings
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    gemini_model_id: str = Field(default="gemini-1.5-flash-latest", env="GEMINI_MODEL_ID")

    # Vault exclusion settings
    excluded_folders: List[str] = Field(
        default=["Journaling", "Atlas/People", "Atlas/work People"],
        env="JARVIS_EXCLUDED_FOLDERS",
        description="Folders to exclude from indexing and querying"
    )
    
    # Vault settings
    vault_path: str = Field(
        default="",
        env="JARVIS_VAULT_PATH",
        description="Path to primary Obsidian vault"
    )
    vault_watch: bool = Field(
        default=True,
        env="JARVIS_VAULT_WATCH",
        description="Enable file system watching"
    )
    
    # Vector database settings
    vector_db_backend: str = Field(
        default="duckdb",
        env="JARVIS_VECTOR_DB_BACKEND",
        description="Vector database backend: duckdb, chroma, pinecone"
    )
    vector_db_path: str = Field(
        default="resources/data/jarvis-vector.duckdb",
        env="JARVIS_VECTOR_DB_PATH",
        description="Path to DuckDB vector database file"
    )
    vector_db_read_only: bool = Field(default=False, env="JARVIS_VECTOR_DB_READ_ONLY")
    
    # ChromaDB settings
    chroma_collection_name: str = Field(default="jarvis-embeddings", env="JARVIS_CHROMA_COLLECTION_NAME")
    chroma_persist_directory: Optional[str] = Field(default=None, env="JARVIS_CHROMA_PERSIST_DIRECTORY")
    chroma_host: Optional[str] = Field(default=None, env="JARVIS_CHROMA_HOST")
    chroma_port: Optional[int] = Field(default=None, env="JARVIS_CHROMA_PORT")
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = Field(default=None, env="JARVIS_PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="JARVIS_PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="jarvis-embeddings", env="JARVIS_PINECONE_INDEX_NAME")
    
    # Graph database settings
    graph_db_backend: str = Field(
        default="neo4j",
        env="JARVIS_GRAPH_DB_BACKEND", 
        description="Graph database backend: neo4j, arangodb"
    )
    graph_enabled: bool = Field(
        default=True,
        env="JARVIS_GRAPH_ENABLED",
        description="Enable graph database integration"
    )
    
    # Neo4j settings
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        env="JARVIS_NEO4J_URI",
        description="Neo4j database URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        env="JARVIS_NEO4J_USER",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="password",
        env="JARVIS_NEO4J_PASSWORD",
        description="Neo4j password"
    )
    
    # ArangoDB settings
    arango_hosts: str = Field(default="http://localhost:8529", env="JARVIS_ARANGO_HOSTS")
    arango_database: str = Field(default="jarvis", env="JARVIS_ARANGO_DATABASE")
    arango_username: str = Field(default="root", env="JARVIS_ARANGO_USERNAME")
    arango_password: Optional[str] = Field(default=None, env="JARVIS_ARANGO_PASSWORD")
    
    # Embedding settings
    embedding_model_name: str = Field(
        default="paraphrase-MiniLM-L6-v2",
        env="JARVIS_EMBEDDING_MODEL_NAME",
        description="Sentence transformer model name"
    )
    embedding_device: str = Field(
        default="mps",  # Apple Silicon optimized
        env="JARVIS_EMBEDDING_DEVICE",
        description="PyTorch device for embeddings"
    )
    embedding_batch_size: int = Field(
        default=32,
        env="JARVIS_EMBEDDING_BATCH_SIZE",
        description="Batch size for embedding generation"
    )
    
    # MCP server settings
    mcp_server_name: str = Field(default="jarvis-assistant", env="JARVIS_MCP_SERVER_NAME")
    mcp_server_version: str = Field(default="0.2.0", env="JARVIS_MCP_SERVER_VERSION")
    mcp_cache_size: int = Field(default=100, env="JARVIS_MCP_CACHE_SIZE", description="Maximum number of cached MCP tool call results.")
    mcp_cache_ttl: int = Field(default=300, env="JARVIS_MCP_CACHE_TTL", description="Time to live for cached MCP entries in seconds.")
    
    # Metrics and monitoring settings
    metrics_enabled: bool = Field(
        default=True,
        env="JARVIS_METRICS_ENABLED", 
        description="Enable performance metrics collection"
    )
    metrics_sampling_rate: float = Field(
        default=1.0,
        env="JARVIS_METRICS_SAMPLING_RATE",
        description="Sampling rate for metrics collection (0.0-1.0)"
    )
    metrics_retention_minutes: int = Field(
        default=60,
        env="JARVIS_METRICS_RETENTION_MINUTES",
        description="How long to retain metrics in memory (minutes)"
    )
    metrics_detailed_logging: bool = Field(
        default=False,
        env="JARVIS_METRICS_DETAILED_LOGGING",
        description="Enable detailed metrics logging for debugging"
    )
    
    # Indexing settings
    index_batch_size: int = Field(
        default=32,
        env="JARVIS_INDEX_BATCH_SIZE",
        description="Batch size for document indexing"
    )
    index_enqueue_all: bool = Field(
        default=False,
        env="JARVIS_INDEX_ENQUEUE_ALL",
        description="Force reindex all documents on startup"
    )
    
    # Search settings
    search_default_limit: int = Field(
        default=10,
        env="JARVIS_SEARCH_DEFAULT_LIMIT",
        description="Default limit for search results"
    )
    search_similarity_threshold: float = Field(
        default=-10.0,
        env="JARVIS_SEARCH_SIMILARITY_THRESHOLD",
        description="Minimum similarity threshold for search results (negative values for cosine distance)"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        env="JARVIS_LOG_LEVEL",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="JARVIS_LOG_FORMAT",
        description="Logging format"
    )
    
    # Service container settings
    use_dependency_injection: bool = Field(
        default=False,
        env="JARVIS_USE_DEPENDENCY_INJECTION",
        description="Enable dependency injection container (experimental)"
    )
    service_logging_enabled: bool = Field(
        default=True,
        env="JARVIS_SERVICE_LOGGING_ENABLED",
        description="Enable detailed service logging"
    )
    service_health_check_interval: int = Field(
        default=60,
        env="JARVIS_SERVICE_HEALTH_CHECK_INTERVAL",
        description="Service health check interval in seconds"
    )
    
    # Extension system settings
    extensions_enabled: bool = Field(
        default=False,
        env="JARVIS_EXTENSIONS_ENABLED",
        description="Enable extension system (Phase 0)"
    )
    extensions_auto_load: List[str] = Field(
        default=[],
        env="JARVIS_EXTENSIONS_AUTO_LOAD",
        description="List of extensions to automatically load on startup"
    )
    extensions_directory: str = Field(
        default="src/jarvis/extensions",
        env="JARVIS_EXTENSIONS_DIRECTORY",
        description="Directory containing extensions"
    )
    extensions_config: Dict[str, Any] = Field(
        default={},
        description="Extension-specific configuration"
    )
    
    # AI Extension settings (Phase 1+)
    ai_extension_enabled: bool = Field(
        default=False,
        env="JARVIS_AI_EXTENSION_ENABLED",
        description="Enable AI extension with LLM capabilities"
    )
    ai_llm_provider: str = Field(
        default="ollama",
        env="JARVIS_AI_LLM_PROVIDER",
        description="LLM provider: ollama, huggingface"
    )
    ai_llm_models: List[str] = Field(
        default=["llama2:7b"],
        env="JARVIS_AI_LLM_MODELS",
        description="Available LLM models"
    )
    ai_max_memory_gb: int = Field(
        default=8,
        env="JARVIS_AI_MAX_MEMORY_GB",
        description="Maximum memory usage for AI operations (GB)"
    )
    ai_timeout_seconds: int = Field(
        default=30,
        env="JARVIS_AI_TIMEOUT_SECONDS",
        description="Timeout for AI operations (seconds)"
    )
    ai_graphrag_enabled: bool = Field(
        default=False,
        env="JARVIS_AI_GRAPHRAG_ENABLED",
        description="Enable GraphRAG capabilities"
    )
    ai_workflows_enabled: bool = Field(
        default=False,
        env="JARVIS_AI_WORKFLOWS_ENABLED",
        description="Enable workflow orchestration"
    )
    
    # Analytics settings
    analytics_enabled: bool = Field(
        default=True,
        env="JARVIS_ANALYTICS_ENABLED",
        description="Enable vault analytics engine"
    )
    analytics_cache_enabled: bool = Field(
        default=True,
        env="JARVIS_ANALYTICS_CACHE_ENABLED",
        description="Enable analytics caching"
    )
    analytics_cache_max_size_mb: int = Field(
        default=100,
        env="JARVIS_ANALYTICS_CACHE_MAX_SIZE_MB",
        description="Maximum analytics cache size in MB"
    )
    analytics_cache_ttl_minutes: int = Field(
        default=60,
        env="JARVIS_ANALYTICS_CACHE_TTL_MINUTES",
        description="Analytics cache time-to-live in minutes"
    )
    analytics_quality_scoring_algorithm: str = Field(
        default="comprehensive",
        env="JARVIS_ANALYTICS_QUALITY_SCORING_ALGORITHM",
        description="Quality scoring algorithm: basic, comprehensive"
    )
    analytics_quality_connection_weight: float = Field(
        default=0.3,
        env="JARVIS_ANALYTICS_QUALITY_CONNECTION_WEIGHT",
        description="Weight for connection metrics in quality scoring (0.0-1.0)"
    )
    analytics_quality_freshness_weight: float = Field(
        default=0.2,
        env="JARVIS_ANALYTICS_QUALITY_FRESHNESS_WEIGHT",
        description="Weight for freshness metrics in quality scoring (0.0-1.0)"
    )
    analytics_domains_clustering_threshold: float = Field(
        default=0.7,
        env="JARVIS_ANALYTICS_DOMAINS_CLUSTERING_THRESHOLD",
        description="Similarity threshold for domain clustering (0.0-1.0)"
    )
    analytics_domains_min_cluster_size: int = Field(
        default=3,
        env="JARVIS_ANALYTICS_DOMAINS_MIN_CLUSTER_SIZE",
        description="Minimum notes required to form a domain cluster"
    )
    analytics_domains_max_domains: int = Field(
        default=20,
        env="JARVIS_ANALYTICS_DOMAINS_MAX_DOMAINS",
        description="Maximum number of domains to identify"
    )
    analytics_max_processing_time_seconds: int = Field(
        default=15,
        env="JARVIS_ANALYTICS_MAX_PROCESSING_TIME_SECONDS",
        description="Maximum processing time for analytics operations"
    )
    analytics_enable_parallel_processing: bool = Field(
        default=True,
        env="JARVIS_ANALYTICS_ENABLE_PARALLEL_PROCESSING",
        description="Enable parallel processing for analytics"
    )
    analytics_sample_large_vaults: bool = Field(
        default=True,
        env="JARVIS_ANALYTICS_SAMPLE_LARGE_VAULTS",
        description="Use sampling for large vaults to improve performance"
    )
    analytics_sample_threshold: int = Field(
        default=5000,
        env="JARVIS_ANALYTICS_SAMPLE_THRESHOLD",
        description="Note count threshold for enabling sampling"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "JARVIS_"
        
    def get_vault_path(self) -> Optional[Path]:
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
    
    def get_analytics_config(self) -> Dict[str, Any]:
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
