"""
Database Factory Pattern for Multi-Backend Support.

This module provides factory methods to create database instances based on
configuration, enabling easy switching between different database backends.
"""


from jarvis.core.interfaces import IGraphDatabase, IVectorDatabase
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ConfigurationError, ServiceError
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class DatabaseConfig:
    """Base configuration for database connections."""

    def __init__(self, backend_type: str, **kwargs):
        self.backend_type = backend_type
        self.config = kwargs

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)


class VectorDatabaseConfig(DatabaseConfig):
    """Configuration for vector database backends."""

    @classmethod
    def from_settings(cls, settings: JarvisSettings) -> "VectorDatabaseConfig":
        """Create configuration from settings."""
        # Default to DuckDB for backward compatibility
        backend_type = getattr(settings, 'vector_db_backend', 'duckdb')

        if backend_type == 'duckdb':
            return cls(
                backend_type='duckdb',
                database_path=settings.get_vector_db_path(),
                read_only=settings.vector_db_read_only
            )
        elif backend_type == 'chroma':
            return cls(
                backend_type='chroma',
                collection_name=getattr(settings, 'chroma_collection_name', 'jarvis-embeddings'),
                persist_directory=getattr(settings, 'chroma_persist_directory', '~/.jarvis/chroma'),
                host=getattr(settings, 'chroma_host', None),
                port=getattr(settings, 'chroma_port', None)
            )
        elif backend_type == 'pinecone':
            return cls(
                backend_type='pinecone',
                api_key=getattr(settings, 'pinecone_api_key', None),
                environment=getattr(settings, 'pinecone_environment', None),
                index_name=getattr(settings, 'pinecone_index_name', 'jarvis-embeddings')
            )
        else:
            raise ConfigurationError(f"Unsupported vector database backend: {backend_type}")


class GraphDatabaseConfig(DatabaseConfig):
    """Configuration for graph database backends."""

    @classmethod
    def from_settings(cls, settings: JarvisSettings) -> "GraphDatabaseConfig":
        """Create configuration from settings."""
        # Default to Neo4j for backward compatibility
        backend_type = getattr(settings, 'graph_db_backend', 'neo4j')

        if backend_type == 'neo4j':
            return cls(
                backend_type='neo4j',
                uri=settings.neo4j_uri,
                username=settings.neo4j_user,
                password=settings.neo4j_password,
                enabled=settings.graph_enabled
            )
        elif backend_type == 'arangodb':
            return cls(
                backend_type='arangodb',
                hosts=getattr(settings, 'arango_hosts', 'http://localhost:8529'),
                database=getattr(settings, 'arango_database', 'jarvis'),
                username=getattr(settings, 'arango_username', 'root'),
                password=getattr(settings, 'arango_password', None)
            )
        else:
            raise ConfigurationError(f"Unsupported graph database backend: {backend_type}")


class DatabaseFactory:
    """Factory for creating database instances based on configuration."""

    # Registry of available database backends
    _vector_backends: dict[str, type[IVectorDatabase]] = {}
    _graph_backends: dict[str, type[IGraphDatabase]] = {}

    @classmethod
    def register_vector_backend(cls, backend_type: str, backend_class: type[IVectorDatabase]):
        """Register a vector database backend."""
        cls._vector_backends[backend_type] = backend_class
        logger.info(f"Registered vector database backend: {backend_type}")

    @classmethod
    def register_graph_backend(cls, backend_type: str, backend_class: type[IGraphDatabase]):
        """Register a graph database backend."""
        cls._graph_backends[backend_type] = backend_class
        logger.info(f"Registered graph database backend: {backend_type}")

    @classmethod
    def create_vector_database(cls, config: VectorDatabaseConfig) -> IVectorDatabase:
        """Create a vector database instance based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Vector database instance implementing IVectorDatabase
            
        Raises:
            ConfigurationError: If backend type is not supported
            ServiceError: If database connection fails
        """
        backend_type = config.backend_type

        if backend_type not in cls._vector_backends:
            available = list(cls._vector_backends.keys())
            raise ConfigurationError(f"Vector database backend '{backend_type}' not registered. Available: {available}")

        backend_class = cls._vector_backends[backend_type]

        try:
            logger.info(f"Creating vector database instance: {backend_type}")
            return backend_class.from_config(config)
        except Exception as e:
            logger.error(f"Failed to create vector database {backend_type}: {e}")
            raise ServiceError(f"Failed to create vector database {backend_type}: {e}") from e

    @classmethod
    def create_graph_database(cls, config: GraphDatabaseConfig) -> IGraphDatabase:
        """Create a graph database instance based on configuration.
        
        Args:
            config: Graph database configuration
            
        Returns:
            Graph database instance implementing IGraphDatabase
            
        Raises:
            ConfigurationError: If backend type is not supported
            ServiceError: If database connection fails
        """
        backend_type = config.backend_type

        if backend_type not in cls._graph_backends:
            available = list(cls._graph_backends.keys())
            raise ConfigurationError(f"Graph database backend '{backend_type}' not registered. Available: {available}")

        backend_class = cls._graph_backends[backend_type]

        try:
            logger.info(f"Creating graph database instance: {backend_type}")
            return backend_class.from_config(config)
        except Exception as e:
            logger.error(f"Failed to create graph database {backend_type}: {e}")
            raise ServiceError(f"Failed to create graph database {backend_type}: {e}") from e

    @classmethod
    def list_available_backends(cls) -> dict[str, list]:
        """List all available database backends."""
        return {
            'vector': list(cls._vector_backends.keys()),
            'graph': list(cls._graph_backends.keys())
        }

    @classmethod
    def validate_config(cls, config: DatabaseConfig) -> bool:
        """Validate database configuration without creating instance."""
        try:
            if isinstance(config, VectorDatabaseConfig):
                backend_type = config.backend_type
                return backend_type in cls._vector_backends
            elif isinstance(config, GraphDatabaseConfig):
                backend_type = config.backend_type
                return backend_type in cls._graph_backends
            return False
        except Exception:
            return False


# Auto-registration of built-in backends
def _register_builtin_backends():
    """Register built-in database backends."""
    try:
        from jarvis.services.vector.database import VectorDatabase
        DatabaseFactory.register_vector_backend('duckdb', VectorDatabase)
    except ImportError:
        logger.warning("DuckDB vector backend not available")

    try:
        from jarvis.services.graph.database import GraphDatabase
        DatabaseFactory.register_graph_backend('neo4j', GraphDatabase)
    except ImportError:
        logger.warning("Neo4j graph backend not available")


# Register built-in backends on module import
_register_builtin_backends()
