"""
Service Discovery Client for easy integration with Service Registry.

This module provides convenient client classes for services to register themselves
and discover other services in the system.
"""

import asyncio
import socket
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from jarvis.core.service_registry import (
    LoadBalancingStrategy,
    ServiceInstance,
    ServiceRegistry,
    ServiceStatus,
    get_service_registry,
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for service registration."""
    name: str
    version: str = "1.0.0"
    endpoint: str | None = None
    health_check_endpoint: str | None = None
    tags: set[str] = None
    metadata: dict[str, Any] = None
    heartbeat_interval: float = 30.0
    auto_register: bool = True
    auto_deregister: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.metadata is None:
            self.metadata = {}
        if self.endpoint is None:
            self.endpoint = self._get_default_endpoint()

    def _get_default_endpoint(self) -> str:
        """Get default endpoint based on hostname and available port."""
        hostname = socket.gethostname()
        port = self._find_free_port()
        return f"http://{hostname}:{port}"

    def _find_free_port(self) -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port


class ServiceClient:
    """Client for interacting with the service registry."""

    def __init__(self, config: ServiceConfig, registry: ServiceRegistry | None = None):
        """Initialize the service client.
        
        Args:
            config: Service configuration
            registry: Optional service registry instance
        """
        self.config = config
        self.registry = registry or get_service_registry()
        self.instance_id = f"{config.name}-{int(time.time() * 1000)}"
        self.instance: ServiceInstance | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._running = False
        self._lock = threading.Lock()

        logger.info(f"Service client initialized for {config.name}")

    async def start(self) -> bool:
        """Start the service client and register with registry.
        
        Returns:
            True if startup successful, False otherwise
        """
        with self._lock:
            if self._running:
                logger.warning(f"Service client for {self.config.name} already running")
                return True

            try:
                # Create service instance
                self.instance = ServiceInstance(
                    id=self.instance_id,
                    name=self.config.name,
                    version=self.config.version,
                    endpoint=self.config.endpoint,
                    metadata=self.config.metadata.copy(),
                    tags=self.config.tags.copy(),
                    status=ServiceStatus.STARTING,
                    health_check_url=self.config.health_check_endpoint
                )

                # Register with service registry
                if self.config.auto_register:
                    success = self.registry.register_service(
                        self.config.name,
                        self.instance
                    )

                    if not success:
                        logger.error(f"Failed to register service {self.config.name}")
                        return False

                # Start heartbeat task
                if self.config.heartbeat_interval > 0:
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Mark as healthy
                if self.instance:
                    self.instance.status = ServiceStatus.HEALTHY
                    self.registry.update_service_health(
                        self.config.name,
                        self.instance_id,
                        ServiceStatus.HEALTHY
                    )

                self._running = True
                logger.info(f"Service client started for {self.config.name}")
                return True

            except Exception as e:
                logger.error(f"Failed to start service client for {self.config.name}: {e}")
                return False

    async def stop(self) -> bool:
        """Stop the service client and deregister from registry.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        with self._lock:
            if not self._running:
                return True

            try:
                self._running = False

                # Stop heartbeat task
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    self._heartbeat_task = None

                # Deregister from service registry
                if self.config.auto_deregister and self.instance:
                    self.registry.deregister_service(
                        self.config.name,
                        self.instance_id
                    )

                logger.info(f"Service client stopped for {self.config.name}")
                return True

            except Exception as e:
                logger.error(f"Failed to stop service client for {self.config.name}: {e}")
                return False

    def update_status(self, status: ServiceStatus, metadata: dict[str, Any] | None = None) -> bool:
        """Update service status.
        
        Args:
            status: New service status
            metadata: Optional metadata update
            
        Returns:
            True if update successful, False otherwise
        """
        if not self.instance:
            return False

        return self.registry.update_service_health(
            self.config.name,
            self.instance_id,
            status,
            metadata
        )

    def update_load_factor(self, load_factor: float) -> bool:
        """Update service load factor.
        
        Args:
            load_factor: New load factor (higher = more loaded)
            
        Returns:
            True if update successful, False otherwise
        """
        if not self.instance:
            return False

        self.instance.load_factor = load_factor
        return True

    def discover_services(
        self,
        service_name: str,
        tags: set[str] | None = None,
        healthy_only: bool = True
    ) -> list[ServiceInstance]:
        """Discover other services.
        
        Args:
            service_name: Name of service to discover
            tags: Optional tags to filter by
            healthy_only: Whether to return only healthy instances
            
        Returns:
            List of matching service instances
        """
        return self.registry.discover_service(service_name, tags, healthy_only)

    def get_service_instance(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        tags: set[str] | None = None
    ) -> ServiceInstance | None:
        """Get a service instance using load balancing.
        
        Args:
            service_name: Name of service
            strategy: Load balancing strategy
            tags: Optional tags to filter by
            
        Returns:
            Selected service instance or None if none available
        """
        return self.registry.get_service_instance(service_name, strategy, tags)

    def subscribe_to_service_changes(
        self,
        service_name: str,
        callback: Callable[[str, str, ServiceInstance], None]
    ) -> str:
        """Subscribe to changes for a service.
        
        Args:
            service_name: Name of service to watch
            callback: Callback function
            
        Returns:
            Subscription ID
        """
        return self.registry.subscribe_to_service_changes(service_name, callback)

    def unsubscribe_from_service_changes(
        self,
        service_name: str,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from service changes.
        
        Args:
            service_name: Name of service
            subscription_id: Subscription ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.registry.unsubscribe_from_service_changes(service_name, subscription_id)

    async def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self._running:
            try:
                if self.instance:
                    self.registry.update_service_health(
                        self.config.name,
                        self.instance_id,
                        ServiceStatus.HEALTHY
                    )

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop for {self.config.name}: {e}")
                await asyncio.sleep(5)  # Brief retry delay

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class ServiceDiscoveryMixin:
    """Mixin class to add service discovery capabilities to existing services."""

    def __init__(self, service_config: ServiceConfig, *args, **kwargs):
        """Initialize the mixin.
        
        Args:
            service_config: Service configuration
        """
        super().__init__(*args, **kwargs)
        self.service_client = ServiceClient(service_config)
        self._service_started = False

    async def start_service_discovery(self) -> bool:
        """Start service discovery capabilities.
        
        Returns:
            True if startup successful, False otherwise
        """
        if self._service_started:
            return True

        success = await self.service_client.start()
        if success:
            self._service_started = True
        return success

    async def stop_service_discovery(self) -> bool:
        """Stop service discovery capabilities.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        if not self._service_started:
            return True

        success = await self.service_client.stop()
        if success:
            self._service_started = False
        return success

    def discover_services(
        self,
        service_name: str,
        tags: set[str] | None = None,
        healthy_only: bool = True
    ) -> list[ServiceInstance]:
        """Discover other services."""
        return self.service_client.discover_services(service_name, tags, healthy_only)

    def get_service_instance(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        tags: set[str] | None = None
    ) -> ServiceInstance | None:
        """Get a service instance using load balancing."""
        return self.service_client.get_service_instance(service_name, strategy, tags)

    def update_service_status(self, status: ServiceStatus, metadata: dict[str, Any] | None = None) -> bool:
        """Update service status."""
        return self.service_client.update_status(status, metadata)

    def update_service_load(self, load_factor: float) -> bool:
        """Update service load factor."""
        return self.service_client.update_load_factor(load_factor)


@asynccontextmanager
async def managed_service(config: ServiceConfig):
    """Context manager for automatic service lifecycle management.
    
    Args:
        config: Service configuration
        
    Yields:
        ServiceClient instance
    """
    client = ServiceClient(config)
    try:
        await client.start()
        yield client
    finally:
        await client.stop()


# Convenience functions
async def register_service(
    name: str,
    version: str = "1.0.0",
    endpoint: str | None = None,
    tags: set[str] | None = None,
    metadata: dict[str, Any] | None = None
) -> ServiceClient:
    """Convenience function to register a service.
    
    Args:
        name: Service name
        version: Service version
        endpoint: Service endpoint
        tags: Optional tags
        metadata: Optional metadata
        
    Returns:
        ServiceClient instance
    """
    config = ServiceConfig(
        name=name,
        version=version,
        endpoint=endpoint,
        tags=tags or set(),
        metadata=metadata or {}
    )

    client = ServiceClient(config)
    await client.start()
    return client


def discover_service(
    service_name: str,
    tags: set[str] | None = None,
    healthy_only: bool = True
) -> list[ServiceInstance]:
    """Convenience function to discover services.
    
    Args:
        service_name: Name of service to discover
        tags: Optional tags to filter by
        healthy_only: Whether to return only healthy instances
        
    Returns:
        List of matching service instances
    """
    registry = get_service_registry()
    return registry.discover_service(service_name, tags, healthy_only)


def get_service(
    service_name: str,
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    tags: set[str] | None = None
) -> ServiceInstance | None:
    """Convenience function to get a service instance.
    
    Args:
        service_name: Name of service
        strategy: Load balancing strategy
        tags: Optional tags to filter by
        
    Returns:
        Selected service instance or None if none available
    """
    registry = get_service_registry()
    return registry.get_service_instance(service_name, strategy, tags)
