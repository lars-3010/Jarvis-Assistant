"""
Integration tests for the dependency injection system.

These tests verify that the entire DI system works together properly,
including service creation, dependency resolution, and lifecycle management.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.core.container import ServiceContainer
from jarvis.mcp.container_context import ContainerAwareMCPServerContext
from jarvis.mcp.server import create_mcp_server
from jarvis.utils.config import JarvisSettings


@pytest.fixture
def test_settings_with_di():
    """Create test settings with dependency injection enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield JarvisSettings(
            vault_path=str(Path(temp_dir) / "vault"),
            vector_db_path=str(Path(temp_dir) / "test.duckdb"),
            use_dependency_injection=True,
            metrics_enabled=True,
            graph_enabled=False  # Disable to avoid Neo4j dependency
        )


@pytest.fixture
def test_settings_without_di():
    """Create test settings with dependency injection disabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield JarvisSettings(
            vault_path=str(Path(temp_dir) / "vault"),
            vector_db_path=str(Path(temp_dir) / "test.duckdb"),
            use_dependency_injection=False,
            metrics_enabled=True,
            graph_enabled=False
        )


@pytest.fixture
def test_vaults(test_settings_with_di):
    """Create test vaults with temporary directories."""
    vault_path = Path(test_settings_with_di.vault_path)
    vault_path.mkdir(parents=True, exist_ok=True)
    
    # Create a simple markdown file
    test_file = vault_path / "test.md"
    test_file.write_text("# Test Note\n\nThis is a test note.")
    
    return {"default": vault_path}


class TestDependencyInjectionIntegration:
    """Integration tests for the dependency injection system."""
    
    def test_service_container_full_lifecycle(self, test_settings_with_di):
        """Test the complete lifecycle of the service container."""
        # Create vault directory
        vault_path = Path(test_settings_with_di.vault_path)
        vault_path.mkdir(parents=True, exist_ok=True)

        container = ServiceContainer(test_settings_with_di)
        
        try:
            # Configure services
            container.configure_default_services()
            
            # Verify services can be retrieved
            info = container.get_service_info()
            assert info["registered_services"] > 0
            
            # Test that services are properly instantiated
            from jarvis.core.interfaces import IVectorEncoder, IMetrics
            
            encoder = container.get(IVectorEncoder)
            assert encoder is not None
            
            # Test metrics if enabled
            if test_settings_with_di.metrics_enabled:
                metrics = container.get(IMetrics)
                assert metrics is not None
                
                # Test metrics functionality
                metrics.record_counter("test_counter", 5)
                metrics_data = metrics.get_metrics()
                assert "test_counter" in metrics_data["metrics"]
        
        finally:
            container.dispose()
    
    @patch('jarvis.services.vector.database.VectorDatabase.__init__', return_value=None)
    @patch('jarvis.services.vector.encoder.VectorEncoder.__init__', return_value=None)
    def test_container_aware_context_creation(self, mock_encoder_init, mock_db_init, 
                                            test_vaults, test_settings_with_di):
        """Test creating a container-aware MCP context."""
        database_path = Path(test_settings_with_di.vector_db_path)
        
        with ContainerAwareMCPServerContext(
            vaults=test_vaults,
            database_path=database_path,
            settings=test_settings_with_di
        ) as context:
            
            # Verify context is properly initialized
            assert context.vaults == test_vaults
            assert context.database_path == database_path
            assert context.settings == test_settings_with_di
            assert context.container is not None
            
            # Verify service info can be retrieved
            info = context.get_service_info()
            assert "container_info" in info
            assert "vault_count" in info
            assert info["vault_count"] == len(test_vaults)
    
    @patch('jarvis.services.vector.database.VectorDatabase')
    @patch('jarvis.services.vector.encoder.VectorEncoder')
    def test_mcp_server_with_dependency_injection(self, mock_encoder, mock_database,
                                                 test_vaults, test_settings_with_di):
        """Test MCP server creation with dependency injection enabled."""
        database_path = Path(test_settings_with_di.vector_db_path)
        
        # Mock the database and encoder classes
        mock_database.return_value = Mock()
        mock_encoder.return_value = Mock()
        
        server = create_mcp_server(
            vaults=test_vaults,
            database_path=database_path,
            settings=test_settings_with_di
        )
        
        assert server is not None
        assert server.name == "jarvis-assistant"
    
    @patch('jarvis.services.vector.database.VectorDatabase')
    @patch('jarvis.services.vector.encoder.VectorEncoder')
    def test_mcp_server_without_dependency_injection(self, mock_encoder, mock_database,
                                                    test_vaults, test_settings_without_di):
        """Test MCP server creation with dependency injection disabled."""
        database_path = Path(test_settings_without_di.vector_db_path)
        
        # Mock the database and encoder classes
        mock_database.return_value = Mock()
        mock_encoder.return_value = Mock()
        
        server = create_mcp_server(
            vaults=test_vaults,
            database_path=database_path,
            settings=test_settings_without_di
        )
        
        assert server is not None
        assert server.name == "jarvis-assistant"
    
    def test_service_container_error_handling(self, test_settings_with_di):
        """Test error handling in the service container."""
        container = ServiceContainer(test_settings_with_di)
        
        # Test getting unregistered service
        from jarvis.core.interfaces import IVectorDatabase
        from jarvis.utils.errors import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            container.get(IVectorDatabase)
    
    @patch('jarvis.core.container.ServiceContainer.configure_default_services')
    def test_graceful_service_failure(self, mock_configure, test_settings_with_di):
        """Test graceful handling of service configuration failures."""
        # Make service configuration fail
        mock_configure.side_effect = ImportError("Service not available")
        
        container = ServiceContainer(test_settings_with_di)
        
        # Should not raise exception during initialization
        try:
            container.configure_default_services()
            pytest.fail("Expected ImportError to be raised")
        except ImportError:
            # This is expected
            pass
    
    def test_container_service_lifecycle(self, test_settings_with_di):
        """Test proper service lifecycle management."""
        services_closed = []
        
        class MockService:
            def __init__(self):
                self.closed = False
            
            def close(self):
                self.closed = True
                services_closed.append(self)
        
        container = ServiceContainer(test_settings_with_di)
        
        # Register mock service
        container.register_instance(MockService, MockService())
        
        # Get the service to ensure it's in singletons
        service = container.get(MockService)
        assert not service.closed
        
        # Dispose container
        container.dispose()
        
        # Verify service was closed
        assert service.closed
        assert len(services_closed) == 1
    
    def test_container_context_manager_cleanup(self, test_vaults, test_settings_with_di):
        """Test proper cleanup when using container as context manager."""
        database_path = Path(test_settings_with_di.vector_db_path)
        
        context = None
        with patch('jarvis.core.container.ServiceContainer.dispose') as mock_dispose:
            with ContainerAwareMCPServerContext(
                vaults=test_vaults,
                database_path=database_path,
                settings=test_settings_with_di
            ) as ctx:
                context = ctx
                assert context is not None
            
            # Verify dispose was called
            mock_dispose.assert_called_once()
    
    def test_service_container_concurrent_access(self, test_settings_with_di):
        """Test that service container handles concurrent access safely."""
        import threading
        import time
        
        container = ServiceContainer(test_settings_with_di)
        
        class SlowService:
            def __init__(self):
                time.sleep(0.1)  # Simulate slow initialization
                self.initialized = True
        
        container.register(SlowService, SlowService, singleton=True)
        
        services = []
        errors = []
        
        def get_service():
            try:
                service = container.get(SlowService)
                services.append(service)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads to access the service simultaneously
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_service)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred and all got the same singleton instance
        assert len(errors) == 0
        assert len(services) == 5
        assert all(service is services[0] for service in services)
        
        container.dispose()