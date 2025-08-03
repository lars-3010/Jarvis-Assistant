"""
Unit tests for the ContainerAwareMCPServerContext.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.mcp.container_context import ContainerAwareMCPServerContext
from jarvis.core.interfaces import IVectorDatabase, IVectorEncoder, IVectorSearcher
from jarvis.utils.config import JarvisSettings


@pytest.fixture
def test_settings():
    """Create test settings."""
    return JarvisSettings(
        vault_path="/tmp/test_vault",
        vector_db_path="/tmp/test.duckdb",
        metrics_enabled=True,
        graph_enabled=True,
        use_dependency_injection=True
    )


@pytest.fixture
def test_vaults():
    """Create test vaults dictionary."""
    return {
        "default": Path("/tmp/test_vault"),
        "secondary": Path("/tmp/test_vault2")
    }


@pytest.fixture
def test_database_path():
    """Create test database path."""
    return Path("/tmp/test.duckdb")


class TestContainerAwareMCPServerContext:
    """Test the ContainerAwareMCPServerContext class."""
    
    @patch('jarvis.core.container.ServiceContainer.configure_default_services')
    def test_context_initialization(self, mock_configure, test_vaults, test_database_path, test_settings):
        """Test context initialization."""
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        assert context.vaults == test_vaults
        assert context.database_path == test_database_path
        assert context.settings == test_settings
        assert context.container is not None
        assert context.ranker is not None
        assert context.mcp_cache is not None
        mock_configure.assert_called_once()
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_database_property(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test database property access."""
        mock_database = Mock()
        mock_get.return_value = mock_database
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        database = context.database
        
        mock_get.assert_called_with(IVectorDatabase)
        assert database == mock_database
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_encoder_property(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test encoder property access."""
        mock_encoder = Mock()
        mock_get.return_value = mock_encoder
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        encoder = context.encoder
        
        mock_get.assert_called_with(IVectorEncoder)
        assert encoder == mock_encoder
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_searcher_property(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test searcher property access."""
        mock_searcher = Mock()
        mock_get.return_value = mock_searcher
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        searcher = context.searcher
        
        mock_get.assert_called_with(IVectorSearcher)
        assert searcher == mock_searcher
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_graph_database_property_success(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test graph database property access when available."""
        mock_graph_db = Mock()
        mock_get.return_value = mock_graph_db
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        graph_db = context.graph_database
        
        assert graph_db == mock_graph_db
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_graph_database_property_failure(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test graph database property access when unavailable."""
        mock_get.side_effect = Exception("Service not available")
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        graph_db = context.graph_database
        
        assert graph_db is None
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_vault_readers_property(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test vault readers property access."""
        mock_reader = Mock()
        mock_get.return_value = mock_reader
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        readers = context.vault_readers
        
        assert len(readers) == len(test_vaults)
        for vault_name in test_vaults.keys():
            assert vault_name in readers
            assert readers[vault_name] == mock_reader
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_health_checker_property(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test health checker property access."""
        mock_health_checker = Mock()
        mock_get.return_value = mock_health_checker
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        health_checker = context.health_checker
        
        assert health_checker == mock_health_checker
    
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_metrics_property_enabled(self, mock_get, test_vaults, test_database_path, test_settings):
        """Test metrics property when enabled."""
        mock_metrics = Mock()
        mock_get.return_value = mock_metrics
        test_settings.metrics_enabled = True
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        metrics = context.metrics
        
        assert metrics == mock_metrics
    
    def test_metrics_property_disabled(self, test_vaults, test_database_path, test_settings):
        """Test metrics property when disabled."""
        test_settings.metrics_enabled = False
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        metrics = context.metrics
        
        assert metrics is None
    
    @patch('jarvis.core.container.ServiceContainer.get_service_info')
    @patch('jarvis.core.container.ServiceContainer.get')
    def test_get_service_info(self, mock_get, mock_get_service_info, test_vaults, test_database_path, test_settings):
        """Test service information retrieval."""
        mock_container_info = {"registered_services": 5}
        mock_get_service_info.return_value = mock_container_info
        
        mock_health_checker = Mock()
        mock_health_checker.get_overall_health.return_value = {"status": "healthy"}
        mock_get.return_value = mock_health_checker
        
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        info = context.get_service_info()
        
        assert "container_info" in info
        assert "vault_count" in info
        assert "database_path" in info
        assert "settings" in info
        assert "health_status" in info
        
        assert info["container_info"] == mock_container_info
        assert info["vault_count"] == len(test_vaults)
        assert info["database_path"] == str(test_database_path)
        assert info["health_status"]["status"] == "healthy"
    
    @patch('jarvis.core.container.ServiceContainer.dispose')
    def test_close(self, mock_dispose, test_vaults, test_database_path, test_settings):
        """Test context cleanup."""
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        # Mock the cache
        context.mcp_cache = Mock()
        
        context.close()
        
        context.mcp_cache.clear.assert_called_once()
        mock_dispose.assert_called_once()
    
    def test_clear_cache(self, test_vaults, test_database_path, test_settings):
        """Test cache clearing."""
        context = ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        )
        
        # Mock the cache
        context.mcp_cache = Mock()
        
        context.clear_cache()
        
        context.mcp_cache.clear.assert_called_once()
    
    @patch('jarvis.core.container.ServiceContainer.dispose')
    def test_context_manager(self, mock_dispose, test_vaults, test_database_path, test_settings):
        """Test context manager behavior."""
        with ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=test_database_path,
            settings=test_settings
        ) as context:
            assert context is not None
            # Mock the cache for testing
            context.mcp_cache = Mock()
        
        # After exiting context, dispose should be called
        mock_dispose.assert_called_once()