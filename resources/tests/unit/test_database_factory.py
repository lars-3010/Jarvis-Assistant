"""
Unit tests for the DatabaseFactory and configuration system.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.database.factory import (
    DatabaseFactory, 
    VectorDatabaseConfig, 
    GraphDatabaseConfig,
    DatabaseConfig
)
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ConfigurationError, ServiceError


class TestDatabaseConfig:
    """Test database configuration classes."""
    
    def test_database_config_creation(self):
        """Test basic DatabaseConfig creation."""
        config = DatabaseConfig("test_backend", param1="value1", param2="value2")
        
        assert config.backend_type == "test_backend"
        assert config.get("param1") == "value1"
        assert config.get("param2") == "value2"
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"


class TestVectorDatabaseConfig:
    """Test vector database configuration."""
    
    def test_duckdb_config_from_settings(self):
        """Test DuckDB configuration creation from settings."""
        settings = JarvisSettings(
            vector_db_backend="duckdb",
            vector_db_path="test.duckdb",
            vector_db_read_only=True
        )
        
        config = VectorDatabaseConfig.from_settings(settings)
        
        assert config.backend_type == "duckdb"
        assert str(config.get("database_path")).endswith("test.duckdb")
        assert config.get("read_only") is True
    
    def test_chroma_config_from_settings(self):
        """Test ChromaDB configuration creation from settings."""
        settings = JarvisSettings(
            vector_db_backend="chroma",
            chroma_collection_name="test-collection",
            chroma_persist_directory="test-dir"
        )
        
        config = VectorDatabaseConfig.from_settings(settings)
        
        assert config.backend_type == "chroma"
        assert config.get("collection_name") == "test-collection"
        assert config.get("persist_directory") == "test-dir"
    
    def test_pinecone_config_from_settings(self):
        """Test Pinecone configuration creation from settings."""
        settings = JarvisSettings(
            vector_db_backend="pinecone",
            pinecone_api_key="test-key",
            pinecone_environment="test-env",
            pinecone_index_name="test-index"
        )
        
        config = VectorDatabaseConfig.from_settings(settings)
        
        assert config.backend_type == "pinecone"
        assert config.get("api_key") == "test-key"
        assert config.get("environment") == "test-env"
        assert config.get("index_name") == "test-index"
    
    def test_unsupported_backend_raises_error(self):
        """Test that unsupported backends raise ConfigurationError."""
        settings = JarvisSettings(vector_db_backend="unsupported")
        
        with pytest.raises(ConfigurationError, match="Unsupported vector database backend"):
            VectorDatabaseConfig.from_settings(settings)


class TestGraphDatabaseConfig:
    """Test graph database configuration."""
    
    def test_neo4j_config_from_settings(self):
        """Test Neo4j configuration creation from settings."""
        settings = JarvisSettings(
            graph_db_backend="neo4j",
            neo4j_uri="bolt://test:7687",
            neo4j_user="test_user",
            neo4j_password="test_pass",
            graph_enabled=True
        )
        
        config = GraphDatabaseConfig.from_settings(settings)
        
        assert config.backend_type == "neo4j"
        assert config.get("uri") == "bolt://test:7687"
        assert config.get("username") == "test_user"
        assert config.get("password") == "test_pass"
        assert config.get("enabled") is True
    
    def test_arangodb_config_from_settings(self):
        """Test ArangoDB configuration creation from settings."""
        settings = JarvisSettings(
            graph_db_backend="arangodb",
            arango_hosts="http://test:8529",
            arango_database="test_db",
            arango_username="test_user",
            arango_password="test_pass"
        )
        
        config = GraphDatabaseConfig.from_settings(settings)
        
        assert config.backend_type == "arangodb"
        assert config.get("hosts") == "http://test:8529"
        assert config.get("database") == "test_db"
        assert config.get("username") == "test_user"
        assert config.get("password") == "test_pass"


class TestDatabaseFactory:
    """Test DatabaseFactory functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear factory registrations
        DatabaseFactory._vector_backends.clear()
        DatabaseFactory._graph_backends.clear()
    
    def test_register_vector_backend(self):
        """Test vector backend registration."""
        self.setUp()
        
        class MockVectorDB:
            pass
        
        DatabaseFactory.register_vector_backend("mock", MockVectorDB)
        
        assert "mock" in DatabaseFactory._vector_backends
        assert DatabaseFactory._vector_backends["mock"] == MockVectorDB
    
    def test_register_graph_backend(self):
        """Test graph backend registration."""
        self.setUp()
        
        class MockGraphDB:
            pass
        
        DatabaseFactory.register_graph_backend("mock", MockGraphDB)
        
        assert "mock" in DatabaseFactory._graph_backends
        assert DatabaseFactory._graph_backends["mock"] == MockGraphDB
    
    def test_create_vector_database_success(self):
        """Test successful vector database creation."""
        self.setUp()
        
        class MockVectorDB:
            @classmethod
            def from_config(cls, config):
                return cls()
        
        DatabaseFactory.register_vector_backend("mock", MockVectorDB)
        config = VectorDatabaseConfig("mock")
        
        db = DatabaseFactory.create_vector_database(config)
        
        assert isinstance(db, MockVectorDB)
    
    def test_create_vector_database_unregistered_backend(self):
        """Test error when creating unregistered vector backend."""
        self.setUp()
        
        config = VectorDatabaseConfig("unregistered")
        
        with pytest.raises(ConfigurationError, match="not registered"):
            DatabaseFactory.create_vector_database(config)
    
    def test_create_vector_database_creation_failure(self):
        """Test error handling when database creation fails."""
        self.setUp()
        
        class MockVectorDB:
            @classmethod
            def from_config(cls, config):
                raise Exception("Creation failed")
        
        DatabaseFactory.register_vector_backend("mock", MockVectorDB)
        config = VectorDatabaseConfig("mock")
        
        with pytest.raises(ServiceError, match="Failed to create vector database"):
            DatabaseFactory.create_vector_database(config)
    
    def test_list_available_backends(self):
        """Test listing available backends."""
        self.setUp()
        
        class MockVectorDB:
            pass
        
        class MockGraphDB:
            pass
        
        DatabaseFactory.register_vector_backend("mock_vector", MockVectorDB)
        DatabaseFactory.register_graph_backend("mock_graph", MockGraphDB)
        
        backends = DatabaseFactory.list_available_backends()
        
        assert "vector" in backends
        assert "graph" in backends
        assert "mock_vector" in backends["vector"]
        assert "mock_graph" in backends["graph"]
    
    def test_validate_config_vector(self):
        """Test config validation for vector databases."""
        self.setUp()
        
        class MockVectorDB:
            pass
        
        DatabaseFactory.register_vector_backend("mock", MockVectorDB)
        
        valid_config = VectorDatabaseConfig("mock")
        invalid_config = VectorDatabaseConfig("unregistered")
        
        assert DatabaseFactory.validate_config(valid_config) is True
        assert DatabaseFactory.validate_config(invalid_config) is False
    
    def test_validate_config_graph(self):
        """Test config validation for graph databases."""
        self.setUp()
        
        class MockGraphDB:
            pass
        
        DatabaseFactory.register_graph_backend("mock", MockGraphDB)
        
        valid_config = GraphDatabaseConfig("mock")
        invalid_config = GraphDatabaseConfig("unregistered")
        
        assert DatabaseFactory.validate_config(valid_config) is True
        assert DatabaseFactory.validate_config(invalid_config) is False


@pytest.mark.integration
class TestDatabaseFactoryIntegration:
    """Integration tests for DatabaseFactory with real backends."""
    
    def test_duckdb_creation_from_factory(self):
        """Test creating DuckDB instance through factory."""
        settings = JarvisSettings(
            vector_db_backend="duckdb",
            vector_db_path="test_factory.duckdb"
        )
        
        config = VectorDatabaseConfig.from_settings(settings)
        
        # This should work if DuckDB backend is registered
        try:
            db = DatabaseFactory.create_vector_database(config)
            assert db is not None
            db.close()
            
            # Clean up
            test_path = Path("test_factory.duckdb")
            if test_path.exists():
                test_path.unlink()
        except Exception as e:
            pytest.skip(f"DuckDB backend not available: {e}")
    
    def test_neo4j_creation_from_factory(self):
        """Test creating Neo4j instance through factory."""
        settings = JarvisSettings(
            graph_db_backend="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            graph_enabled=True
        )
        
        config = GraphDatabaseConfig.from_settings(settings)
        
        # This will only work if Neo4j is running
        try:
            db = DatabaseFactory.create_graph_database(config)
            assert db is not None
            db.close()
        except Exception as e:
            pytest.skip(f"Neo4j backend not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])