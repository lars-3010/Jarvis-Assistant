"""
Service Registry Pattern for Dynamic Service Discovery.

This module provides a comprehensive service registry that enables dynamic service
discovery, health monitoring, and load balancing for distributed service management.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.core.interfaces import IMetrics
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceInstance:
    """Represents a service instance in the registry."""
    id: str
    name: str
    version: str
    endpoint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    status: ServiceStatus = ServiceStatus.STARTING
    last_heartbeat: float = field(default_factory=time.time)
    health_check_url: str | None = None
    load_factor: float = 1.0
    registration_time: float = field(default_factory=time.time)
    failure_count: int = 0

    def is_healthy(self, heartbeat_timeout: float = 30.0) -> bool:
        """Check if service instance is healthy."""
        if self.status != ServiceStatus.HEALTHY:
            return False

        time_since_heartbeat = time.time() - self.last_heartbeat
        return time_since_heartbeat < heartbeat_timeout

    def record_failure(self) -> None:
        """Record a failure for this service instance."""
        self.failure_count += 1
        if self.failure_count >= 3:
            self.status = ServiceStatus.UNHEALTHY

    def record_success(self) -> None:
        """Record a success for this service instance."""
        self.failure_count = 0
        self.status = ServiceStatus.HEALTHY
        self.last_heartbeat = time.time()


@dataclass
class ServiceDefinition:
    """Defines a service type with its requirements."""
    name: str
    version_requirement: str = "*"
    required_tags: set[str] = field(default_factory=set)
    load_balancing_strategy: str = "round_robin"
    health_check_interval: float = 30.0
    max_failures: int = 3


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"


class ServiceRegistry:
    """Central registry for service discovery and management."""

    def __init__(self, metrics: IMetrics | None = None):
        """Initialize the service registry.
        
        Args:
            metrics: Optional metrics collector
        """
        self.metrics = metrics
        self._services: dict[str, dict[str, ServiceInstance]] = {}
        self._service_definitions: dict[str, ServiceDefinition] = {}
        self._load_balancer_state: dict[str, int] = {}
        self._subscribers: dict[str, list[Callable]] = {}
        self._health_check_tasks: dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=10)

        logger.info("Service registry initialized")

    def register_service(
        self,
        service_name: str,
        instance: ServiceInstance,
        replace_existing: bool = False
    ) -> bool:
        """Register a service instance.
        
        Args:
            service_name: Name of the service
            instance: Service instance to register
            replace_existing: Whether to replace existing instance with same ID
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                if service_name not in self._services:
                    self._services[service_name] = {}

                service_instances = self._services[service_name]

                if instance.id in service_instances and not replace_existing:
                    logger.warning(f"Service instance {instance.id} already exists for {service_name}")
                    return False

                service_instances[instance.id] = instance

                # Initialize load balancer state if needed
                if service_name not in self._load_balancer_state:
                    self._load_balancer_state[service_name] = 0

                # Record metrics
                if self.metrics:
                    self.metrics.record_counter(
                        "service_registry.registrations",
                        tags={"service": service_name, "instance": instance.id}
                    )

                # Start health checking for this instance
                self._start_health_check(service_name, instance)

                # Notify subscribers
                self._notify_subscribers(service_name, "registered", instance)

                logger.info(f"Registered service instance {instance.id} for {service_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to register service {service_name}: {e}")
                return False

    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance.
        
        Args:
            service_name: Name of the service
            instance_id: ID of the instance to deregister
            
        Returns:
            True if deregistration successful, False otherwise
        """
        with self._lock:
            try:
                if service_name not in self._services:
                    return False

                service_instances = self._services[service_name]
                if instance_id not in service_instances:
                    return False

                instance = service_instances[instance_id]
                instance.status = ServiceStatus.STOPPING

                # Stop health checking
                self._stop_health_check(service_name, instance_id)

                # Remove from registry
                del service_instances[instance_id]

                # Clean up empty service entries
                if not service_instances:
                    del self._services[service_name]
                    if service_name in self._load_balancer_state:
                        del self._load_balancer_state[service_name]

                # Record metrics
                if self.metrics:
                    self.metrics.record_counter(
                        "service_registry.deregistrations",
                        tags={"service": service_name, "instance": instance_id}
                    )

                # Notify subscribers
                self._notify_subscribers(service_name, "deregistered", instance)

                logger.info(f"Deregistered service instance {instance_id} for {service_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to deregister service {service_name}: {e}")
                return False

    def discover_service(
        self,
        service_name: str,
        tags: set[str] | None = None,
        healthy_only: bool = True
    ) -> list[ServiceInstance]:
        """Discover service instances.
        
        Args:
            service_name: Name of the service to discover
            tags: Optional tags to filter by
            healthy_only: Whether to return only healthy instances
            
        Returns:
            List of matching service instances
        """
        with self._lock:
            if service_name not in self._services:
                return []

            instances = list(self._services[service_name].values())

            # Filter by health
            if healthy_only:
                instances = [inst for inst in instances if inst.is_healthy()]

            # Filter by tags
            if tags:
                instances = [
                    inst for inst in instances
                    if tags.issubset(inst.tags)
                ]

            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "service_registry.discoveries",
                    tags={"service": service_name}
                )

            return instances

    def get_service_instance(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        tags: set[str] | None = None
    ) -> ServiceInstance | None:
        """Get a service instance using load balancing strategy.
        
        Args:
            service_name: Name of the service
            strategy: Load balancing strategy to use
            tags: Optional tags to filter by
            
        Returns:
            Selected service instance or None if none available
        """
        instances = self.discover_service(service_name, tags=tags, healthy_only=True)

        if not instances:
            return None

        if len(instances) == 1:
            return instances[0]

        # Apply load balancing strategy
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, instances)
        elif strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(instances)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.load_factor)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(service_name, instances)

        return instances[0]

    def update_service_health(
        self,
        service_name: str,
        instance_id: str,
        status: ServiceStatus,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Update service health status.
        
        Args:
            service_name: Name of the service
            instance_id: ID of the instance
            status: New status
            metadata: Optional metadata update
            
        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            try:
                if service_name not in self._services:
                    return False

                service_instances = self._services[service_name]
                if instance_id not in service_instances:
                    return False

                instance = service_instances[instance_id]
                old_status = instance.status
                instance.status = status
                instance.last_heartbeat = time.time()

                if metadata:
                    instance.metadata.update(metadata)

                # Record metrics
                if self.metrics:
                    self.metrics.record_counter(
                        "service_registry.health_updates",
                        tags={
                            "service": service_name,
                            "instance": instance_id,
                            "old_status": old_status.value,
                            "new_status": status.value
                        }
                    )

                # Notify subscribers if status changed
                if old_status != status:
                    self._notify_subscribers(service_name, "status_changed", instance)

                return True

            except Exception as e:
                logger.error(f"Failed to update service health {service_name}: {e}")
                return False

    def subscribe_to_service_changes(
        self,
        service_name: str,
        callback: Callable[[str, str, ServiceInstance], None]
    ) -> str:
        """Subscribe to service changes.
        
        Args:
            service_name: Name of the service to watch
            callback: Callback function (service_name, event_type, instance)
            
        Returns:
            Subscription ID
        """
        with self._lock:
            if service_name not in self._subscribers:
                self._subscribers[service_name] = []

            subscription_id = str(uuid.uuid4())
            self._subscribers[service_name].append((subscription_id, callback))

            logger.debug(f"Added subscription {subscription_id} for service {service_name}")
            return subscription_id

    def unsubscribe_from_service_changes(
        self,
        service_name: str,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from service changes.
        
        Args:
            service_name: Name of the service
            subscription_id: Subscription ID to remove
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        with self._lock:
            if service_name not in self._subscribers:
                return False

            subscribers = self._subscribers[service_name]
            original_count = len(subscribers)

            self._subscribers[service_name] = [
                (sub_id, callback) for sub_id, callback in subscribers
                if sub_id != subscription_id
            ]

            success = len(self._subscribers[service_name]) < original_count

            if success:
                logger.debug(f"Removed subscription {subscription_id} for service {service_name}")

            return success

    def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service registry statistics.
        
        Returns:
            Dictionary with service statistics
        """
        with self._lock:
            stats = {
                "total_services": len(self._services),
                "total_instances": sum(len(instances) for instances in self._services.values()),
                "healthy_instances": 0,
                "unhealthy_instances": 0,
                "services": {}
            }

            for service_name, instances in self._services.items():
                service_stats = {
                    "instance_count": len(instances),
                    "healthy_count": 0,
                    "unhealthy_count": 0,
                    "instances": {}
                }

                for instance_id, instance in instances.items():
                    is_healthy = instance.is_healthy()
                    if is_healthy:
                        service_stats["healthy_count"] += 1
                        stats["healthy_instances"] += 1
                    else:
                        service_stats["unhealthy_count"] += 1
                        stats["unhealthy_instances"] += 1

                    service_stats["instances"][instance_id] = {
                        "status": instance.status.value,
                        "healthy": is_healthy,
                        "last_heartbeat": instance.last_heartbeat,
                        "failure_count": instance.failure_count,
                        "load_factor": instance.load_factor
                    }

                stats["services"][service_name] = service_stats

            return stats

    def _round_robin_select(self, service_name: str, instances: list[ServiceInstance]) -> ServiceInstance:
        """Round robin load balancing selection."""
        current_index = self._load_balancer_state.get(service_name, 0)
        selected_instance = instances[current_index % len(instances)]
        self._load_balancer_state[service_name] = (current_index + 1) % len(instances)
        return selected_instance

    def _weighted_round_robin_select(self, service_name: str, instances: list[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin load balancing selection."""
        total_weight = sum(1.0 / max(inst.load_factor, 0.1) for inst in instances)
        target_weight = (time.time() * total_weight) % total_weight

        current_weight = 0.0
        for instance in instances:
            current_weight += 1.0 / max(instance.load_factor, 0.1)
            if current_weight >= target_weight:
                return instance

        return instances[0]

    def _notify_subscribers(self, service_name: str, event_type: str, instance: ServiceInstance):
        """Notify subscribers of service changes."""
        if service_name not in self._subscribers:
            return

        for subscription_id, callback in self._subscribers[service_name]:
            try:
                callback(service_name, event_type, instance)
            except Exception as e:
                logger.error(f"Error in service change callback {subscription_id}: {e}")

    def _start_health_check(self, service_name: str, instance: ServiceInstance):
        """Start health checking for a service instance."""
        if not instance.health_check_url:
            return

        task_key = f"{service_name}:{instance.id}"
        if task_key in self._health_check_tasks:
            return

        async def health_check_loop():
            """Health check loop for service instance."""
            import aiohttp

            while self._running:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            instance.health_check_url,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                instance.record_success()
                            else:
                                instance.record_failure()
                except Exception as e:
                    logger.debug(f"Health check failed for {service_name}:{instance.id}: {e}")
                    instance.record_failure()

                await asyncio.sleep(30)  # Health check interval

        try:
            task = asyncio.create_task(health_check_loop())
            self._health_check_tasks[task_key] = task
            logger.debug(f"Started health check for {service_name}:{instance.id}")
        except Exception as e:
            logger.warning(f"Failed to start health check for {service_name}:{instance.id}: {e}")

    def _stop_health_check(self, service_name: str, instance_id: str):
        """Stop health checking for a service instance."""
        task_key = f"{service_name}:{instance_id}"
        if task_key in self._health_check_tasks:
            task = self._health_check_tasks[task_key]
            task.cancel()
            del self._health_check_tasks[task_key]
            logger.debug(f"Stopped health check for {service_name}:{instance_id}")

    def start(self):
        """Start the service registry."""
        self._running = True
        logger.info("Service registry started")

    def stop(self):
        """Stop the service registry."""
        self._running = False

        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()
        self._health_check_tasks.clear()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("Service registry stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global service registry instance
_global_service_registry: ServiceRegistry | None = None


def get_service_registry(metrics: IMetrics | None = None) -> ServiceRegistry:
    """Get the global service registry instance.
    
    Args:
        metrics: Optional metrics collector
        
    Returns:
        Global service registry instance
    """
    global _global_service_registry
    if _global_service_registry is None:
        _global_service_registry = ServiceRegistry(metrics)
    return _global_service_registry


def reset_service_registry() -> None:
    """Reset the global service registry (mainly for testing)."""
    global _global_service_registry
    if _global_service_registry:
        _global_service_registry.stop()
    _global_service_registry = None
