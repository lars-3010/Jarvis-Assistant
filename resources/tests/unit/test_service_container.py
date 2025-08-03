"""
Unit tests for the ServiceContainer dependency injection system.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

from jarvis.core.container import ServiceContainer, ServiceRegistration
from jarvis.core.interfaces import IVectorDatabase, IVectorEncoder, IHealthChecker
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ConfigurationError


# Mock implementations for testing
class MockVectorDatabase:
    """Mock vector database for testing."""
    
    def __init__(self, database_path: Path, read_only: bool = False):
        self.database_path = database_path
        self.read_only = read_only
        self.closed = False
    
    def close(self):
        self.closed = True
    
    def num_notes(self) -> int:
        return 42


class MockVectorEncoder:
    """Mock vector encoder for testing."""
    
    def __init__(self):
        self.initialized = True
    
    def encode(self, text: str):
        return f"encoded_{text}"


class MockHealthChecker:
    """Mock health checker for testing."""
    
    def __init__(self, settings: JarvisSettings):
        self.settings = settings
    
    def get_overall_health(self) -> Dict[str, Any]:
        return {"status": "healthy", "mock": True}


class MockDependentService:
    """Mock service that depends on other services."""
    
    def __init__(self, database: IVectorDatabase, encoder: IVectorEncoder):
        self.database = database
        self.encoder = encoder


@pytest.fixture
def test_settings():
    """Create test settings."""
    return JarvisSettings(
        vault_path="/tmp/test_vault",
        vector_db_path="/tmp/test.duckdb"
    )


@pytest.fixture
def container(test_settings):
    """Create a service container for testing."""
    return ServiceContainer(test_settings)


class TestServiceContainer:
    """Test the ServiceContainer class."""
    
    def test_container_initialization(self, test_settings):
        """Test container initialization."""
        container = ServiceContainer(test_settings)
        
        assert container.settings == test_settings
        assert len(container._registrations) == 0
        assert len(container._singletons) == 0
    
    def test_service_registration(self, container):
        """Test service registration."""
        container.register(IVectorDatabase, MockVectorDatabase, singleton=True)
        
        assert IVectorDatabase in container._registrations
        registration = container._registrations[IVectorDatabase]
        assert registration.interface == IVectorDatabase
        assert registration.implementation == MockVectorDatabase
        assert registration.singleton is True
    
    def test_service_retrieval(self, container):
        """Test service retrieval."""
        container.register(IVectorEncoder, MockVectorEncoder, singleton=True)
        
        service = container.get(IVectorEncoder)
        
        assert isinstance(service, MockVectorEncoder)
        assert service.initialized is True
    
    def test_singleton_behavior(self, container):
        """Test singleton behavior."""
        container.register(IVectorEncoder, MockVectorEncoder, singleton=True)
        
        service1 = container.get(IVectorEncoder)
        service2 = container.get(IVectorEncoder)
        
        assert service1 is service2
    
    def test_non_singleton_behavior(self, container):
        """Test non-singleton behavior."""
        container.register(IVectorEncoder, MockVectorEncoder, singleton=False)
        
        service1 = container.get(IVectorEncoder)
        service2 = container.get(IVectorEncoder)
        
        assert service1 is not service2
        assert isinstance(service1, MockVectorEncoder)
        assert isinstance(service2, MockVectorEncoder)
    
    def test_dependency_injection(self, container):
        """Test automatic dependency injection."""
        container.register(IVectorDatabase, MockVectorDatabase, singleton=True)
        container.register(IVectorEncoder, MockVectorEncoder, singleton=True)
        
        # Register a service that depends on the other two
        container.register(MockDependentService, MockDependentService, singleton=True)
        
        service = container.get(MockDependentService)
        
        assert isinstance(service, MockDependentService)
        assert isinstance(service.database, MockVectorDatabase)
        assert isinstance(service.encoder, MockVectorEncoder)
    
    def test_unregistered_service_error(self, container):
        """Test error when requesting unregistered service."""
        with pytest.raises(ConfigurationError, match="Service .* is not registered"):
            container.get(IVectorDatabase)
    
    def test_circular_dependency_detection(self, container):
        """Test circular dependency detection."""
        # Create classes that would cause circular dependency
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
        
        container.register(ServiceA, ServiceA)
        container.register(ServiceB, ServiceB)
        
        with pytest.raises(ConfigurationError, match="Circular dependency detected"):
            container.get(ServiceA)
    
    def test_instance_registration(self, container):
        """Test registering an existing instance."""
        instance = MockVectorEncoder()
        container.register_instance(IVectorEncoder, instance)
        
        retrieved = container.get(IVectorEncoder)
        assert retrieved is instance
    
    def test_factory_registration(self, container):
        """Test factory function registration."""
        def encoder_factory():
            encoder = MockVectorEncoder()
            encoder.factory_created = True
            return encoder
        
        container.register(IVectorEncoder, MockVectorEncoder, factory=encoder_factory)
        
        service = container.get(IVectorEncoder)
        assert isinstance(service, MockVectorEncoder)
        assert hasattr(service, 'factory_created')
        assert service.factory_created is True
    
    def test_configure_default_services(self, container):
        """Test default service configuration."""
        # This test might fail if actual services aren't available
        # In a real environment, we'd need to mock the imports
        try:
            container.configure_default_services()
            info = container.get_service_info()
            assert info["registered_services"] > 0
        except ImportError:
            pytest.skip("Real services not available in test environment")
    
    def test_service_info(self, container):
        """Test service information retrieval."""
        container.register(IVectorEncoder, MockVectorEncoder, singleton=True)
        container.register(IVectorDatabase, MockVectorDatabase, singleton=False)
        
        # Create an instance for one service
        container.get(IVectorEncoder)
        
        info = container.get_service_info()
        
        assert info["registered_services"] == 2
        assert info["singleton_instances"] == 1
        assert "IVectorEncoder" in info["services"]
        assert "IVectorDatabase" in info["services"]
        assert info["services"]["IVectorEncoder"]["has_instance"] is True
        assert info["services"]["IVectorDatabase"]["has_instance"] is False
    
    def test_dispose(self, container):
        """Test container disposal."""
        container.register(IVectorDatabase, MockVectorDatabase, singleton=True)
        
        service = container.get(IVectorDatabase)
        assert not service.closed
        
        container.dispose()
        
        assert service.closed
        assert len(container._singletons) == 0
        assert len(container._registrations) == 0
    
    def test_context_manager(self, test_settings):
        """Test context manager behavior."""
        database_service = None
        
        with ServiceContainer(test_settings) as container:
            container.register(IVectorDatabase, MockVectorDatabase, singleton=True)
            database_service = container.get(IVectorDatabase)
            assert not database_service.closed
        
        # After exiting context, service should be closed
        assert database_service.closed


class TestServiceRegistration:
    """Test the ServiceRegistration class."""
    
    def test_registration_creation(self):
        """Test service registration creation."""
        registration = ServiceRegistration(
            interface=IVectorEncoder,
            implementation=MockVectorEncoder,
            singleton=True,
            factory=None
        )
        
        assert registration.interface == IVectorEncoder
        assert registration.implementation == MockVectorEncoder
        assert registration.singleton is True
        assert registration.factory is None
    
    def test_registration_with_factory(self):
        """Test service registration with factory."""
        def test_factory():
            return MockVectorEncoder()
        
        registration = ServiceRegistration(
            interface=IVectorEncoder,
            implementation=MockVectorEncoder,
            singleton=False,
            factory=test_factory
        )
        
        assert registration.factory == test_factory
        assert registration.singleton is False