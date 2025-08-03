"""
Dependency injection container for Jarvis Assistant.

This module provides a service container that manages dependencies and service lifecycle,
enabling loose coupling and better testability.
"""

import inspect
from typing import Any, Dict, Type, TypeVar, Callable, Optional, List
from pathlib import Path
from jarvis.utils.config import JarvisSettings
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import ConfigurationError
from jarvis.core.interfaces import IVectorDatabase, IGraphDatabase

logger = setup_logging(__name__)

T = TypeVar('T')


class ServiceContainer:
    """Dependency injection container for managing services and their dependencies."""
    
    def __init__(self, settings: JarvisSettings):
        """Initialize the service container.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._registrations: Dict[Type, 'ServiceRegistration'] = {}
        self._building: set = set()  # Track services being built to prevent cycles
        
    def register(
        self,
        interface: Type[T],
        implementation: Type[T],
        singleton: bool = True,
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a service with the container.
        
        Args:
            interface: The interface/abstract class
            implementation: The concrete implementation
            singleton: Whether to treat as singleton
            factory: Optional factory function for custom instantiation
        """
        logger.debug(f"Registering {interface.__name__} -> {implementation.__name__}")
        
        self._registrations[interface] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            singleton=singleton,
            factory=factory
        )
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register an existing instance.
        
        Args:
            interface: The interface/abstract class
            instance: The instance to register
        """
        logger.debug(f"Registering instance for {interface.__name__}")
        self._singletons[interface] = instance
    
    def get(self, interface: Type[T]) -> T:
        """Get a service instance.
        
        Args:
            interface: The interface/abstract class to get
            
        Returns:
            Service instance
            
        Raises:
            ConfigurationError: If service is not registered or circular dependency detected
        """
        logger.debug(f"🔍 Requesting service: {interface.__name__}")
        
        # Check if we're already building this service (circular dependency)
        if interface in self._building:
            logger.error(f"❌ Circular dependency detected for {interface.__name__}")
            raise ConfigurationError(f"Circular dependency detected for {interface.__name__}")
        
        # Check if it's a singleton and already exists
        if interface in self._singletons:
            logger.debug(f"✅ Returning existing singleton for {interface.__name__}")
            return self._singletons[interface]
        
        # Check if it's registered
        if interface not in self._registrations:
            logger.error(f"❌ Service {interface.__name__} is not registered")
            logger.debug(f"📋 Available registrations: {list(self._registrations.keys())}")
            raise ConfigurationError(f"Service {interface.__name__} is not registered")
        
        registration = self._registrations[interface]
        logger.debug(f"📋 Found registration for {interface.__name__} -> {registration.implementation.__name__}")
        
        # Mark as building
        self._building.add(interface)
        logger.debug(f"🔧 Building service: {interface.__name__}")
        
        try:
            # Create instance
            if registration.factory:
                logger.debug(f"🏭 Using factory for {interface.__name__}")
                instance = registration.factory()
            else:
                logger.debug(f"🔨 Creating instance using constructor for {interface.__name__}")
                instance = self._create_instance(registration.implementation)
            
            # Store as singleton if configured
            if registration.singleton:
                logger.debug(f"💾 Storing as singleton: {interface.__name__}")
                self._singletons[interface] = instance
            
            logger.info(f"✅ Successfully created instance of {interface.__name__}")
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create instance of {interface.__name__}: {e}")
            raise
        finally:
            # Remove from building set
            self._building.discard(interface)
            logger.debug(f"🏁 Finished building service: {interface.__name__}")
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create an instance with dependency injection.
        
        Args:
            implementation: The class to instantiate
            
        Returns:
            Instance with dependencies injected
        """
        # Get constructor signature
        signature = inspect.signature(implementation.__init__)
        parameters = signature.parameters
        
        # Build arguments
        args = {}
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue
                
            # Check if it's a known dependency
            if param.annotation in self._registrations or param.annotation in self._singletons:
                args[param_name] = self.get(param.annotation)
            elif param_name == 'settings' and isinstance(self.settings, JarvisSettings):
                args[param_name] = self.settings
            elif param_name == 'database_path' and hasattr(self.settings, 'get_vector_db_path'):
                args[param_name] = self.settings.get_vector_db_path()
            elif param_name == 'vaults' and hasattr(self.settings, 'get_vault_path'):
                vault_path = self.settings.get_vault_path()
                args[param_name] = {"default": vault_path} if vault_path else {}
            elif param_name == 'read_only':
                args[param_name] = getattr(self.settings, 'vector_db_read_only', False)
            elif param.default is not inspect.Parameter.empty:
                # Has default value, skip
                continue
            else:
                logger.warning(f"Unknown dependency: {param_name} ({param.annotation})")
        
        # Create instance
        return implementation(**args)
    
    def configure_default_services(self) -> None:
        """Configure default service registrations."""
        logger.info("🔧 Configuring default services")
        
        from jarvis.core.interfaces import (
            IVectorDatabase, IGraphDatabase, IVaultReader,
            IVectorEncoder, IVectorSearcher, IHealthChecker, IMetrics
        )
        from jarvis.services.vector.database import VectorDatabase
        from jarvis.services.vector.encoder import VectorEncoder
        from jarvis.services.vector.searcher import VectorSearcher
        from jarvis.services.graph.database import GraphDatabase
        from jarvis.services.vault.reader import VaultReader
        from jarvis.services.health import HealthChecker
        from jarvis.monitoring.metrics import JarvisMetrics
        
        # Register core services using factory pattern for databases
        logger.debug("📊 Registering vector database")
        try:
            self._register_vector_database()
            logger.debug("✅ Vector database registration completed")
        except Exception as e:
            logger.error(f"❌ Vector database registration failed: {e}")
            raise
        
        logger.debug("🔗 Registering graph database")
        try:
            self._register_graph_database()
            logger.debug("✅ Graph database registration completed")
        except Exception as e:
            logger.warning(f"⚠️ Graph database registration failed: {e}")
        
        # Other services
        logger.debug("🔧 Registering core services")
        try:
            self.register(IVectorEncoder, VectorEncoder, singleton=True)
            logger.debug("✅ VectorEncoder registered")
            
            self.register(IVectorSearcher, VectorSearcher, singleton=True)
            logger.debug("✅ VectorSearcher registered")
            
            self.register(IHealthChecker, HealthChecker, singleton=True)
            logger.debug("✅ HealthChecker registered")
        except Exception as e:
            logger.error(f"❌ Core services registration failed: {e}")
            raise
        
        # Register metrics if enabled
        logger.debug(f"📈 Metrics enabled: {self.settings.metrics_enabled}")
        if self.settings.metrics_enabled:
            try:
                self.register(IMetrics, JarvisMetrics, singleton=True)
                logger.debug("✅ Metrics service registered")
            except Exception as e:
                logger.warning(f"⚠️ Metrics service registration failed: {e}")
        
        # Register vault readers for each vault
        vault_path = self.settings.get_vault_path()
        logger.debug(f"📂 Vault path from settings: {vault_path}")
        
        if vault_path:
            try:
                logger.debug(f"🔧 Creating VaultReader for path: {vault_path}")
                vault_reader = VaultReader(str(vault_path))
                self.register_instance(IVaultReader, vault_reader)
                logger.debug("✅ VaultReader registered successfully")
            except Exception as e:
                logger.error(f"❌ VaultReader registration failed: {e}")
                raise
        else:
            logger.warning("⚠️ No vault path configured, VaultReader not registered")
        
        # Log summary of registered services
        registered_services = list(self._registrations.keys()) + list(self._singletons.keys())
        logger.info(f"🎉 Service configuration completed. Total services: {len(registered_services)}")
        logger.debug(f"📋 Registered services: {[svc.__name__ for svc in registered_services]}")
    
    def _register_vector_database(self) -> None:
        """Register vector database using factory pattern."""
        try:
            from jarvis.database.factory import DatabaseFactory, VectorDatabaseConfig
            
            def vector_db_factory():
                config = VectorDatabaseConfig.from_settings(self.settings)
                return DatabaseFactory.create_vector_database(config)
            
            self.register(IVectorDatabase, VectorDatabase, factory=vector_db_factory, singleton=True)
            logger.info(f"Vector database factory registered: {self.settings.vector_db_backend}")
        except Exception as e:
            logger.warning(f"Failed to register vector database factory: {e}")
            # Fallback to direct registration
            from jarvis.services.vector.database import VectorDatabase
            self.register(IVectorDatabase, VectorDatabase, singleton=True)
    
    def _register_graph_database(self) -> None:
        """Register graph database using factory pattern."""
        try:
            from jarvis.database.factory import DatabaseFactory, GraphDatabaseConfig
            
            def graph_db_factory():
                config = GraphDatabaseConfig.from_settings(self.settings)
                return DatabaseFactory.create_graph_database(config)
            
            self.register(IGraphDatabase, factory=graph_db_factory, singleton=True)
            logger.info(f"Graph database factory registered: {self.settings.graph_db_backend}")
        except Exception as e:
            logger.warning(f"Failed to register graph database factory: {e}")
            # Fallback to direct registration
            from jarvis.services.graph.database import GraphDatabase
            self.register(IGraphDatabase, GraphDatabase, singleton=True)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about registered services.
        
        Returns:
            Dictionary with service information
        """
        info = {
            "registered_services": len(self._registrations),
            "singleton_instances": len(self._singletons),
            "services": {}
        }
        
        for interface, registration in self._registrations.items():
            info["services"][interface.__name__] = {
                "implementation": registration.implementation.__name__,
                "singleton": registration.singleton,
                "has_instance": interface in self._singletons
            }
        
        return info
    
    def dispose(self) -> None:
        """Dispose of all services and clean up resources."""
        logger.info("Disposing service container")
        
        # Close singleton services that have a close method
        for interface, instance in self._singletons.items():
            if hasattr(instance, 'close'):
                try:
                    instance.close()
                    logger.debug(f"Closed service {interface.__name__}")
                except Exception as e:
                    logger.error(f"Error closing service {interface.__name__}: {e}")
        
        # Clear all registrations and instances
        self._singletons.clear()
        self._registrations.clear()
        self._building.clear()
        
        logger.info("Service container disposed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.dispose()


class ServiceRegistration:
    """Represents a service registration in the container."""
    
    def __init__(
        self,
        interface: Type,
        implementation: Type,
        singleton: bool = True,
        factory: Optional[Callable] = None
    ):
        self.interface = interface
        self.implementation = implementation
        self.singleton = singleton
        self.factory = factory