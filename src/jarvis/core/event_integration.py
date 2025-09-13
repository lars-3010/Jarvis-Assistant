"""
Event Integration for connecting events with other system components.

This module provides event handlers and integrations that connect the event system
with service registry, metrics, and other architectural components.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from jarvis.core.events import (
    Event,
    EventFilter,
    EventPriority,
    get_event_bus,
    publish_event,
)
from jarvis.core.interfaces import IMetrics
from jarvis.core.service_registry import (
    ServiceInstance,
    ServiceRegistry,
    ServiceStatus,
    get_service_registry,
)
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


# Standard Event Types
class EventTypes:
    """Standard event types used across the system."""

    # Service events
    SERVICE_REGISTERED = "service.registered"
    SERVICE_DEREGISTERED = "service.deregistered"
    SERVICE_HEALTH_CHANGED = "service.health_changed"
    SERVICE_DISCOVERY_REQUEST = "service.discovery_request"

    # Data events
    VAULT_INDEXED = "vault.indexed"
    VAULT_UPDATED = "vault.updated"
    SEARCH_PERFORMED = "search.performed"
    DOCUMENT_ADDED = "document.added"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPING = "system.stopping"
    SYSTEM_ERROR = "system.error"
    CACHE_CLEARED = "cache.cleared"
    CONFIG_UPDATED = "config.updated"

    # MCP events
    MCP_TOOL_CALLED = "mcp.tool_called"
    MCP_CLIENT_CONNECTED = "mcp.client_connected"
    MCP_CLIENT_DISCONNECTED = "mcp.client_disconnected"

    # Performance events
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance.threshold_exceeded"
    RESOURCE_USAGE_HIGH = "resource.usage_high"


@dataclass
class ServiceEventData:
    """Data structure for service-related events."""
    service_name: str
    instance_id: str
    endpoint: str
    status: str
    metadata: dict[str, Any]
    tags: set[str]


@dataclass
class VaultEventData:
    """Data structure for vault-related events."""
    vault_name: str
    path: str
    operation: str
    metadata: dict[str, Any]
    timestamp: float


@dataclass
class SearchEventData:
    """Data structure for search-related events."""
    query: str
    search_type: str
    vault_name: str | None
    result_count: int
    duration: float
    timestamp: float


class ServiceEventHandler:
    """Event handler for service registry integration."""

    def __init__(self, service_registry: ServiceRegistry | None = None):
        """Initialize the service event handler.
        
        Args:
            service_registry: Optional service registry instance
        """
        self.service_registry = service_registry or get_service_registry()
        self.event_bus = get_event_bus()
        self._subscription_ids: list[str] = []

        logger.info("Service event handler initialized")

    async def start(self):
        """Start the service event handler."""
        # Subscribe to service-related events
        service_filter = EventFilter(
            event_types={
                EventTypes.SERVICE_REGISTERED,
                EventTypes.SERVICE_DEREGISTERED,
                EventTypes.SERVICE_HEALTH_CHANGED,
                EventTypes.SERVICE_DISCOVERY_REQUEST
            }
        )

        subscription_id = self.event_bus.subscribe(
            self._handle_service_event,
            service_filter,
            is_async=True
        )
        self._subscription_ids.append(subscription_id)

        # Subscribe to service registry changes
        self.service_registry.subscribe_to_service_changes(
            "*",  # All services
            self._on_service_registry_change
        )

        logger.info("Service event handler started")

    async def stop(self):
        """Stop the service event handler."""
        for subscription_id in self._subscription_ids:
            self.event_bus.unsubscribe(subscription_id)
        self._subscription_ids.clear()

        logger.info("Service event handler stopped")

    async def _handle_service_event(self, event: Event):
        """Handle service-related events."""
        try:
            if event.event_type == EventTypes.SERVICE_DISCOVERY_REQUEST:
                await self._handle_discovery_request(event)
            elif event.event_type == EventTypes.SERVICE_REGISTERED:
                await self._handle_service_registered(event)
            elif event.event_type == EventTypes.SERVICE_DEREGISTERED:
                await self._handle_service_deregistered(event)
            elif event.event_type == EventTypes.SERVICE_HEALTH_CHANGED:
                await self._handle_service_health_changed(event)

        except Exception as e:
            logger.error(f"Error handling service event {event.event_id}: {e}")

    async def _handle_discovery_request(self, event: Event):
        """Handle service discovery requests."""
        data = event.data
        service_name = data.get('service_name')
        tags = data.get('tags', set())

        if service_name:
            instances = self.service_registry.discover_service(
                service_name,
                tags=tags,
                healthy_only=True
            )

            # Publish discovery response
            await publish_event(
                "service.discovery_response",
                {
                    'request_id': event.event_id,
                    'service_name': service_name,
                    'instances': [inst.to_dict() for inst in instances]
                },
                source="service_event_handler",
                correlation_id=event.correlation_id
            )

    async def _handle_service_registered(self, event: Event):
        """Handle service registration events."""
        logger.info(f"Service registered: {event.data.get('service_name')}")

    async def _handle_service_deregistered(self, event: Event):
        """Handle service deregistration events."""
        logger.info(f"Service deregistered: {event.data.get('service_name')}")

    async def _handle_service_health_changed(self, event: Event):
        """Handle service health change events."""
        data = event.data
        service_name = data.get('service_name')
        status = data.get('status')

        if status == ServiceStatus.UNHEALTHY.value:
            logger.warning(f"Service {service_name} became unhealthy")

            # Could trigger alert or recovery actions here
            await publish_event(
                "system.alert",
                {
                    'alert_type': 'service_unhealthy',
                    'service_name': service_name,
                    'instance_id': data.get('instance_id'),
                    'timestamp': time.time()
                },
                source="service_event_handler",
                priority=EventPriority.HIGH
            )

    def _on_service_registry_change(self, service_name: str, event_type: str, instance: ServiceInstance):
        """Handle service registry changes."""
        try:
            # Convert service registry changes to events
            event_data = {
                'service_name': service_name,
                'instance_id': instance.id,
                'endpoint': instance.endpoint,
                'status': instance.status.value,
                'metadata': instance.metadata,
                'tags': list(instance.tags)
            }

            if event_type == "registered":
                event_type_name = EventTypes.SERVICE_REGISTERED
            elif event_type == "deregistered":
                event_type_name = EventTypes.SERVICE_DEREGISTERED
            elif event_type == "status_changed":
                event_type_name = EventTypes.SERVICE_HEALTH_CHANGED
            else:
                return

            # Publish event asynchronously
            asyncio.create_task(
                publish_event(
                    event_type_name,
                    event_data,
                    source="service_registry"
                )
            )

        except Exception as e:
            logger.error(f"Error handling service registry change: {e}")


class VaultEventHandler:
    """Event handler for vault-related operations."""

    def __init__(self, metrics: IMetrics | None = None):
        """Initialize the vault event handler.
        
        Args:
            metrics: Optional metrics collector
        """
        self.metrics = metrics
        self.event_bus = get_event_bus()
        self._subscription_ids: list[str] = []

        logger.info("Vault event handler initialized")

    async def start(self):
        """Start the vault event handler."""
        vault_filter = EventFilter(
            event_types={
                EventTypes.VAULT_INDEXED,
                EventTypes.VAULT_UPDATED,
                EventTypes.DOCUMENT_ADDED,
                EventTypes.DOCUMENT_UPDATED,
                EventTypes.DOCUMENT_DELETED
            }
        )

        subscription_id = self.event_bus.subscribe(
            self._handle_vault_event,
            vault_filter,
            is_async=True
        )
        self._subscription_ids.append(subscription_id)

        logger.info("Vault event handler started")

    async def stop(self):
        """Stop the vault event handler."""
        for subscription_id in self._subscription_ids:
            self.event_bus.unsubscribe(subscription_id)
        self._subscription_ids.clear()

        logger.info("Vault event handler stopped")

    async def _handle_vault_event(self, event: Event):
        """Handle vault-related events."""
        try:
            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "vault.events",
                    tags={
                        "event_type": event.event_type,
                        "vault_name": event.data.get('vault_name', 'unknown')
                    }
                )

            if event.event_type == EventTypes.VAULT_INDEXED:
                await self._handle_vault_indexed(event)
            elif event.event_type == EventTypes.DOCUMENT_ADDED:
                await self._handle_document_added(event)
            elif event.event_type == EventTypes.DOCUMENT_UPDATED:
                await self._handle_document_updated(event)
            elif event.event_type == EventTypes.DOCUMENT_DELETED:
                await self._handle_document_deleted(event)

        except Exception as e:
            logger.error(f"Error handling vault event {event.event_id}: {e}")

    async def _handle_vault_indexed(self, event: Event):
        """Handle vault indexing completion."""
        data = event.data
        vault_name = data.get('vault_name')
        document_count = data.get('document_count', 0)
        duration = data.get('duration', 0)

        logger.info(f"Vault {vault_name} indexed: {document_count} documents in {duration:.2f}s")

        # Clear relevant caches
        await publish_event(
            EventTypes.CACHE_CLEARED,
            {
                'cache_type': 'search_cache',
                'vault_name': vault_name,
                'reason': 'vault_reindexed'
            },
            source="vault_event_handler"
        )

    async def _handle_document_added(self, event: Event):
        """Handle document addition."""
        data = event.data
        vault_name = data.get('vault_name')
        path = data.get('path')

        logger.debug(f"Document added: {vault_name}:{path}")

    async def _handle_document_updated(self, event: Event):
        """Handle document updates."""
        data = event.data
        vault_name = data.get('vault_name')
        path = data.get('path')

        logger.debug(f"Document updated: {vault_name}:{path}")

    async def _handle_document_deleted(self, event: Event):
        """Handle document deletion."""
        data = event.data
        vault_name = data.get('vault_name')
        path = data.get('path')

        logger.debug(f"Document deleted: {vault_name}:{path}")


class SearchEventHandler:
    """Event handler for search-related operations."""

    def __init__(self, metrics: IMetrics | None = None):
        """Initialize the search event handler.
        
        Args:
            metrics: Optional metrics collector
        """
        self.metrics = metrics
        self.event_bus = get_event_bus()
        self._subscription_ids: list[str] = []
        self._search_stats: dict[str, dict[str, Any]] = {}

        logger.info("Search event handler initialized")

    async def start(self):
        """Start the search event handler."""
        search_filter = EventFilter(
            event_types={EventTypes.SEARCH_PERFORMED}
        )

        subscription_id = self.event_bus.subscribe(
            self._handle_search_event,
            search_filter,
            is_async=True
        )
        self._subscription_ids.append(subscription_id)

        logger.info("Search event handler started")

    async def stop(self):
        """Stop the search event handler."""
        for subscription_id in self._subscription_ids:
            self.event_bus.unsubscribe(subscription_id)
        self._subscription_ids.clear()

        logger.info("Search event handler stopped")

    async def _handle_search_event(self, event: Event):
        """Handle search events."""
        try:
            data = event.data
            search_type = data.get('search_type')
            duration = data.get('duration', 0)
            result_count = data.get('result_count', 0)
            vault_name = data.get('vault_name', 'unknown')

            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "search.queries",
                    tags={
                        "search_type": search_type,
                        "vault_name": vault_name
                    }
                )

                self.metrics.record_histogram(
                    "search.duration",
                    duration,
                    tags={
                        "search_type": search_type,
                        "vault_name": vault_name
                    }
                )

                self.metrics.record_histogram(
                    "search.result_count",
                    result_count,
                    tags={
                        "search_type": search_type,
                        "vault_name": vault_name
                    }
                )

            # Update search statistics
            if search_type not in self._search_stats:
                self._search_stats[search_type] = {
                    'total_queries': 0,
                    'total_duration': 0,
                    'total_results': 0,
                    'avg_duration': 0,
                    'avg_results': 0
                }

            stats = self._search_stats[search_type]
            stats['total_queries'] += 1
            stats['total_duration'] += duration
            stats['total_results'] += result_count
            stats['avg_duration'] = stats['total_duration'] / stats['total_queries']
            stats['avg_results'] = stats['total_results'] / stats['total_queries']

            # Check for performance issues
            if duration > 5.0:  # 5 second threshold
                await publish_event(
                    EventTypes.PERFORMANCE_THRESHOLD_EXCEEDED,
                    {
                        'metric': 'search_duration',
                        'value': duration,
                        'threshold': 5.0,
                        'search_type': search_type,
                        'vault_name': vault_name,
                        'query': data.get('query', '')[:100]  # Truncate for privacy
                    },
                    source="search_event_handler",
                    priority=EventPriority.HIGH
                )

        except Exception as e:
            logger.error(f"Error handling search event {event.event_id}: {e}")

    def get_search_stats(self) -> dict[str, dict[str, Any]]:
        """Get search statistics.
        
        Returns:
            Dictionary with search statistics by type
        """
        return self._search_stats.copy()


class CacheEventHandler:
    """Event handler for cache-related events (metrics, observability)."""

    def __init__(self, metrics: IMetrics | None = None):
        self.metrics = metrics
        self.event_bus = get_event_bus()
        self._subscription_ids: list[str] = []

    async def start(self):
        cache_filter = EventFilter(event_types={EventTypes.CACHE_CLEARED})
        sub_id = self.event_bus.subscribe(
            self._handle_cache_event,
            cache_filter,
            is_async=True,
        )
        self._subscription_ids.append(sub_id)

    async def stop(self):
        for sub_id in self._subscription_ids:
            self.event_bus.unsubscribe(sub_id)
        self._subscription_ids.clear()

    async def _handle_cache_event(self, event: Event):
        try:
            data = event.data or {}
            cache_type = data.get("cache_type", "unknown")
            reason = data.get("reason", "unspecified")
            vault_name = data.get("vault_name", "*")

            if self.metrics:
                self.metrics.record_counter(
                    "analytics.cache.invalidations",
                    value=1,
                    tags={
                        "cache_type": cache_type,
                        "reason": reason,
                        "vault_name": vault_name,
                    },
                )
        except Exception:
            # Metrics should never break core functionality
            pass


class EventIntegrationManager:
    """Manager for all event integrations."""

    def __init__(
        self,
        service_registry: ServiceRegistry | None = None,
        metrics: IMetrics | None = None
    ):
        """Initialize the event integration manager.
        
        Args:
            service_registry: Optional service registry instance
            metrics: Optional metrics collector
        """
        self.service_event_handler = ServiceEventHandler(service_registry)
        self.vault_event_handler = VaultEventHandler(metrics)
        self.search_event_handler = SearchEventHandler(metrics)
        self.cache_event_handler = CacheEventHandler(metrics)
        self._running = False

        logger.info("Event integration manager initialized")

    async def start(self):
        """Start all event handlers."""
        if self._running:
            return

        await self.service_event_handler.start()
        await self.vault_event_handler.start()
        await self.search_event_handler.start()
        await self.cache_event_handler.start()

        self._running = True
        logger.info("Event integration manager started")

    async def stop(self):
        """Stop all event handlers."""
        if not self._running:
            return

        await self.service_event_handler.stop()
        await self.vault_event_handler.stop()
        await self.search_event_handler.stop()
        await self.cache_event_handler.stop()

        self._running = False
        logger.info("Event integration manager stopped")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics from all handlers.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "running": self._running,
            "search_stats": self.search_event_handler.get_search_stats(),
            "event_bus_stats": get_event_bus().get_stats()
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global event integration manager
_global_integration_manager: EventIntegrationManager | None = None


def get_event_integration_manager(
    service_registry: ServiceRegistry | None = None,
    metrics: IMetrics | None = None
) -> EventIntegrationManager:
    """Get the global event integration manager.
    
    Args:
        service_registry: Optional service registry instance
        metrics: Optional metrics collector
        
    Returns:
        Global event integration manager instance
    """
    global _global_integration_manager
    if _global_integration_manager is None:
        _global_integration_manager = EventIntegrationManager(service_registry, metrics)
    return _global_integration_manager


def reset_event_integration_manager() -> None:
    """Reset the global event integration manager (mainly for testing)."""
    global _global_integration_manager
    if _global_integration_manager:
        asyncio.create_task(_global_integration_manager.stop())
    _global_integration_manager = None


# Convenience functions for publishing common events
async def publish_service_event(
    event_type: str,
    service_name: str,
    instance_id: str,
    **kwargs
) -> bool:
    """Publish a service-related event."""
    data = {
        'service_name': service_name,
        'instance_id': instance_id,
        **kwargs
    }

    return await publish_event(
        event_type,
        data,
        source="service_system"
    )


async def publish_vault_event(
    event_type: str,
    vault_name: str,
    path: str,
    operation: str,
    **kwargs
) -> bool:
    """Publish a vault-related event."""
    data = {
        'vault_name': vault_name,
        'path': path,
        'operation': operation,
        'timestamp': time.time(),
        **kwargs
    }

    return await publish_event(
        event_type,
        data,
        source="vault_system"
    )


async def publish_search_event(
    query: str,
    search_type: str,
    result_count: int,
    duration: float,
    vault_name: str | None = None,
    **kwargs
) -> bool:
    """Publish a search-related event."""
    data = {
        'query': query,
        'search_type': search_type,
        'result_count': result_count,
        'duration': duration,
        'vault_name': vault_name,
        'timestamp': time.time(),
        **kwargs
    }

    return await publish_event(
        EventTypes.SEARCH_PERFORMED,
        data,
        source="search_system"
    )
