"""
Event-Driven Architecture for Reactive Systems.

This module provides a comprehensive event system for publishing and subscribing
to events, enabling loose coupling and reactive behavior across services.
"""

import asyncio
import threading
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jarvis.core.interfaces import IMetrics
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Represents an event in the system."""
    event_type: str
    data: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str | None = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    causation_id: str | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'source': self.source,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            data=data['data'],
            timestamp=data['timestamp'],
            source=data.get('source'),
            priority=EventPriority(data.get('priority', EventPriority.NORMAL.value)),
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id'),
            version=data.get('version', 1)
        )

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class EventFilter:
    """Filter for event subscriptions."""

    def __init__(
        self,
        event_types: set[str] | None = None,
        sources: set[str] | None = None,
        min_priority: EventPriority | None = None,
        metadata_filters: dict[str, Any] | None = None
    ):
        self.event_types = event_types or set()
        self.sources = sources or set()
        self.min_priority = min_priority
        self.metadata_filters = metadata_filters or {}

    def matches(self, event: Event) -> bool:
        """Check if event matches filter criteria."""
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check source
        if self.sources and event.source not in self.sources:
            return False

        # Check priority
        if self.min_priority and event.priority.value < self.min_priority.value:
            return False

        # Check metadata filters
        for key, value in self.metadata_filters.items():
            if key not in event.metadata or event.metadata[key] != value:
                return False

        return True


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    subscription_id: str
    handler: Callable[[Event], Any]
    filter: EventFilter
    is_async: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    dead_letter_enabled: bool = True
    created_at: float = field(default_factory=time.time)

    async def handle_event(self, event: Event) -> bool:
        """Handle an event with retry logic.
        
        Returns:
            True if handled successfully, False otherwise
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                if self.is_async:
                    if asyncio.iscoroutinefunction(self.handler):
                        await self.handler(event)
                    else:
                        await asyncio.get_event_loop().run_in_executor(None, self.handler, event)
                else:
                    self.handler(event)

                return True

            except Exception as e:
                last_error = e
                retries += 1

                if retries <= self.max_retries:
                    logger.warning(f"Event handler failed (attempt {retries}): {e}")
                    await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff
                else:
                    logger.error(f"Event handler failed after {self.max_retries} retries: {e}")

        return False


class IEventStore(ABC):
    """Abstract interface for event storage."""

    @abstractmethod
    async def store_event(self, event: Event) -> bool:
        """Store an event."""
        pass

    @abstractmethod
    async def get_events(
        self,
        event_types: set[str] | None = None,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """Retrieve events."""
        pass

    @abstractmethod
    async def get_event_by_id(self, event_id: str) -> Event | None:
        """Get a specific event by ID."""
        pass


class MemoryEventStore(IEventStore):
    """In-memory event store implementation."""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: list[Event] = []
        self._events_by_id: dict[str, Event] = {}
        self._lock = asyncio.Lock()

    async def store_event(self, event: Event) -> bool:
        """Store an event in memory."""
        async with self._lock:
            try:
                self._events.append(event)
                self._events_by_id[event.event_id] = event

                # Maintain size limit
                if len(self._events) > self.max_events:
                    oldest_event = self._events.pop(0)
                    del self._events_by_id[oldest_event.event_id]

                return True
            except Exception as e:
                logger.error(f"Failed to store event {event.event_id}: {e}")
                return False

    async def get_events(
        self,
        event_types: set[str] | None = None,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """Retrieve events from memory."""
        async with self._lock:
            events = self._events.copy()

        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if from_timestamp:
            events = [e for e in events if e.timestamp >= from_timestamp]

        if to_timestamp:
            events = [e for e in events if e.timestamp <= to_timestamp]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    async def get_event_by_id(self, event_id: str) -> Event | None:
        """Get a specific event by ID."""
        async with self._lock:
            return self._events_by_id.get(event_id)


class EventBus:
    """Central event bus for publishing and subscribing to events."""

    def __init__(
        self,
        event_store: IEventStore | None = None,
        metrics: IMetrics | None = None,
        max_concurrent_handlers: int = 100
    ):
        """Initialize the event bus.
        
        Args:
            event_store: Optional event store for persistence
            metrics: Optional metrics collector
            max_concurrent_handlers: Maximum concurrent event handlers
        """
        self.event_store = event_store or MemoryEventStore()
        self.metrics = metrics
        self.max_concurrent_handlers = max_concurrent_handlers

        self._subscriptions: dict[str, EventSubscription] = {}
        self._subscriptions_by_type: dict[str, set[str]] = {}
        self._event_queue: asyncio.Queue | None = None
        self._dead_letter_queue: asyncio.Queue | None = None
        self._processor_task: asyncio.Task | None = None
        self._dead_letter_task: asyncio.Task | None = None
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent_handlers)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready: threading.Event | None = None

        logger.info("Event bus initialized")

    async def start(self):
        """Start the event bus."""
        if self._running:
            return

        self._running = True
        # Capture the running loop for thread-safe publishing
        try:
            self._loop = asyncio.get_running_loop()
            # Queues bound to this loop
            self._event_queue = asyncio.Queue()
            self._dead_letter_queue = asyncio.Queue()
            self._processor_task = asyncio.create_task(self._process_events())
            self._dead_letter_task = asyncio.create_task(self._process_dead_letters())
            logger.info("Event bus started (asyncio loop)")
        except RuntimeError:
            # No running asyncio loop (e.g., Trio). Start a dedicated background loop.
            def _run_loop():
                loop = asyncio.new_event_loop()
                self._loop = loop
                asyncio.set_event_loop(loop)
                # Create queues in this loop
                self._event_queue = asyncio.Queue()
                self._dead_letter_queue = asyncio.Queue()
                self._processor_task = loop.create_task(self._process_events())
                self._dead_letter_task = loop.create_task(self._process_dead_letters())
                if self._loop_ready:
                    self._loop_ready.set()
                loop.run_forever()

            self._loop_ready = threading.Event()
            self._thread = threading.Thread(target=_run_loop, name="EventBusLoop", daemon=True)
            self._thread.start()
            # Wait briefly for loop to start
            self._loop_ready.wait(timeout=1.0)
            logger.info("Event bus started (dedicated background loop)")

    async def stop(self):
        """Stop the event bus."""
        if not self._running:
            return

        self._running = False

        # If running in a dedicated background loop (thread), stop it
        if self._thread and self._loop and not self._loop.is_closed():
            def _cancel_and_stop():
                if self._processor_task:
                    self._processor_task.cancel()
                if self._dead_letter_task:
                    self._dead_letter_task.cancel()
                try:
                    self._loop.stop()
                except Exception:
                    pass

            self._loop.call_soon_threadsafe(_cancel_and_stop)
            self._thread.join(timeout=2.0)
            self._thread = None
            self._loop = None
            logger.info("Event bus stopped (background loop)")
        else:
            # Normal asyncio loop cancellation
            if self._processor_task:
                self._processor_task.cancel()
                try:
                    await self._processor_task
                except asyncio.CancelledError:
                    pass

            if self._dead_letter_task:
                self._dead_letter_task.cancel()
                try:
                    await self._dead_letter_task
                except asyncio.CancelledError:
                    pass

            logger.info("Event bus stopped")
            self._loop = None

    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus.
        
        Args:
            event: Event to publish
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self._running:
            logger.warning("Event bus not running, cannot publish event")
            return False

        # If running on a different loop (background thread), publish thread-safely
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._loop and current_loop is not self._loop:
            try:
                if self.event_store:
                    asyncio.run_coroutine_threadsafe(self.event_store.store_event(event), self._loop)
                if self._event_queue is not None:
                    asyncio.run_coroutine_threadsafe(self._event_queue.put(event), self._loop)

                if self.metrics:
                    self.metrics.record_counter(
                        "event_bus.events_published",
                        tags={"event_type": event.event_type, "priority": event.priority.name},
                    )
                return True
            except Exception as e:
                logger.error(f"Failed cross-loop publish for event {event.event_id}: {e}")
                return False

        # Same-loop async publish
        try:
            # Store event if event store is configured
            if self.event_store:
                await self.event_store.store_event(event)

            # Add to processing queue
            if self._event_queue is None:
                self._event_queue = asyncio.Queue()
            await self._event_queue.put(event)

            # Record metrics
            if self.metrics:
                self.metrics.record_counter(
                    "event_bus.events_published",
                    tags={
                        "event_type": event.event_type,
                        "priority": event.priority.name
                    }
                )

            logger.debug(f"Published event {event.event_id} of type {event.event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False

    def subscribe(
        self,
        handler: Callable[[Event], Any],
        event_filter: EventFilter | None = None,
        is_async: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """Subscribe to events.
        
        Args:
            handler: Event handler function
            event_filter: Optional filter for events
            is_async: Whether handler is async
            max_retries: Maximum retry attempts for failed handlers
            retry_delay: Base delay between retries
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        filter = event_filter or EventFilter()

        subscription = EventSubscription(
            subscription_id=subscription_id,
            handler=handler,
            filter=filter,
            is_async=is_async,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

        self._subscriptions[subscription_id] = subscription

        # Index by event types for efficient lookup
        for event_type in filter.event_types or ["*"]:  # "*" means all types
            if event_type not in self._subscriptions_by_type:
                self._subscriptions_by_type[event_type] = set()
            self._subscriptions_by_type[event_type].add(subscription_id)

        logger.debug(f"Added subscription {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]

        # Remove from type index
        for event_type in subscription.filter.event_types or ["*"]:
            if event_type in self._subscriptions_by_type:
                self._subscriptions_by_type[event_type].discard(subscription_id)
                if not self._subscriptions_by_type[event_type]:
                    del self._subscriptions_by_type[event_type]

        del self._subscriptions[subscription_id]

        logger.debug(f"Removed subscription {subscription_id}")
        return True

    def get_subscriptions(self) -> list[str]:
        """Get all subscription IDs.
        
        Returns:
            List of subscription IDs
        """
        return list(self._subscriptions.keys())

    async def get_events(
        self,
        event_types: set[str] | None = None,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """Get stored events.
        
        Args:
            event_types: Optional event types to filter
            from_timestamp: Optional start timestamp
            to_timestamp: Optional end timestamp
            limit: Optional maximum number of events
            
        Returns:
            List of events
        """
        if not self.event_store:
            return []

        return await self.event_store.get_events(
            event_types=event_types,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            limit=limit
        )

    async def get_event_by_id(self, event_id: str) -> Event | None:
        """Get a specific event by ID.
        
        Args:
            event_id: Event ID to retrieve
            
        Returns:
            Event instance or None if not found
        """
        if not self.event_store:
            return None

        return await self.event_store.get_event_by_id(event_id)

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "running": self._running,
            "total_subscriptions": len(self._subscriptions),
            "subscriptions_by_type": {
                event_type: len(sub_ids)
                for event_type, sub_ids in self._subscriptions_by_type.items()
            },
            "queue_size": self._event_queue.qsize() if self._event_queue else 0,
            "dead_letter_queue_size": self._dead_letter_queue.qsize() if self._dead_letter_queue else 0,
            "max_concurrent_handlers": self.max_concurrent_handlers
        }

    def publish_threadsafe(self, event: Event) -> bool:
        """Publish an event from a non-async thread safely.

        Uses the captured event loop from start() to enqueue the event.
        Returns False if the bus isn't running yet.
        """
        try:
            if not self._running or self._loop is None:
                # Bus not started yet; skip quietly
                return False

            # Store event if event store is configured
            if self.event_store:
                fut_store = asyncio.run_coroutine_threadsafe(
                    self.event_store.store_event(event), self._loop
                )
                # Fire-and-forget; do not block
                _ = fut_store

            # Enqueue for processing
            fut_queue = asyncio.run_coroutine_threadsafe(
                self._event_queue.put(event), self._loop
            )
            _ = fut_queue

            # Metrics (best-effort)
            if self.metrics:
                self.metrics.record_counter(
                    "event_bus.events_published", tags={"event_type": event.event_type}
                )
            return True
        except Exception as e:
            logger.debug(f"Thread-safe publish failed: {e}")
            return False

    async def _process_events(self):
        """Background event processing loop."""
        while self._running:
            try:
                # Wait for event with timeout to allow clean shutdown
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Process event concurrently
                await self._handle_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)

    async def _handle_event(self, event: Event):
        """Handle a single event by dispatching to subscribers."""
        # Find matching subscriptions
        matching_subscriptions = set()

        # Check specific event type subscriptions
        if event.event_type in self._subscriptions_by_type:
            matching_subscriptions.update(self._subscriptions_by_type[event.event_type])

        # Check wildcard subscriptions
        if "*" in self._subscriptions_by_type:
            matching_subscriptions.update(self._subscriptions_by_type["*"])

        if not matching_subscriptions:
            return

        # Create handler tasks
        handler_tasks = []
        for sub_id in matching_subscriptions:
            subscription = self._subscriptions.get(sub_id)
            if subscription and subscription.filter.matches(event):
                task = asyncio.create_task(self._execute_handler(subscription, event))
                handler_tasks.append(task)

        if handler_tasks:
            # Wait for all handlers to complete
            results = await asyncio.gather(*handler_tasks, return_exceptions=True)

            # Record metrics
            if self.metrics:
                success_count = sum(1 for r in results if r is True)
                failure_count = len(results) - success_count

                self.metrics.record_counter(
                    "event_bus.handlers_executed",
                    value=len(results),
                    tags={"event_type": event.event_type}
                )

                if failure_count > 0:
                    self.metrics.record_counter(
                        "event_bus.handler_failures",
                        value=failure_count,
                        tags={"event_type": event.event_type}
                    )

    async def _execute_handler(self, subscription: EventSubscription, event: Event) -> bool:
        """Execute a single event handler with concurrency control."""
        async with self._semaphore:
            success = await subscription.handle_event(event)

            if not success and subscription.dead_letter_enabled:
                await self._dead_letter_queue.put((event, subscription))

            return success

    async def _process_dead_letters(self):
        """Process events that failed handling."""
        while self._running:
            try:
                try:
                    event, subscription = await asyncio.wait_for(
                        self._dead_letter_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                logger.warning(
                    f"Event {event.event_id} failed handling by subscription "
                    f"{subscription.subscription_id}"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_counter(
                        "event_bus.dead_letters",
                        tags={"event_type": event.event_type}
                    )

                # Could implement dead letter storage or notification here

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dead letter processing: {e}")
                await asyncio.sleep(1)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global event bus instance
_global_event_bus: EventBus | None = None


def get_event_bus(
    event_store: IEventStore | None = None,
    metrics: IMetrics | None = None
) -> EventBus:
    """Get the global event bus instance.
    
    Args:
        event_store: Optional event store
        metrics: Optional metrics collector
        
    Returns:
        Global event bus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus(event_store, metrics)
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_event_bus
    if _global_event_bus:
        asyncio.create_task(_global_event_bus.stop())
    _global_event_bus = None


# Convenience functions
async def publish_event(
    event_type: str,
    data: dict[str, Any],
    source: str | None = None,
    priority: EventPriority = EventPriority.NORMAL,
    correlation_id: str | None = None
) -> bool:
    """Convenience function to publish an event.
    
    Args:
        event_type: Type of event
        data: Event data
        source: Optional event source
        priority: Event priority
        correlation_id: Optional correlation ID
        
    Returns:
        True if published successfully, False otherwise
    """
    event = Event(
        event_type=event_type,
        data=data,
        source=source,
        priority=priority,
        correlation_id=correlation_id
    )

    event_bus = get_event_bus()
    return await event_bus.publish(event)


def publish_event_threadsafe(
    event_type: str,
    data: dict[str, Any],
    source: str | None = None,
    priority: EventPriority = EventPriority.NORMAL,
    correlation_id: str | None = None,
) -> bool:
    """Publish an event from a non-async context if the bus is running.

    Returns False if the bus isn't running yet (safe no-op).
    """
    event = Event(
        event_type=event_type,
        data=data,
        source=source,
        priority=priority,
        correlation_id=correlation_id,
    )
    event_bus = get_event_bus()
    return event_bus.publish_threadsafe(event)


def subscribe_to_events(
    event_types: set[str] | None = None,
    handler: Callable[[Event], Any] | None = None,
    is_async: bool = True
):
    """Decorator for subscribing to events.
    
    Args:
        event_types: Optional event types to filter
        handler: Optional handler function
        is_async: Whether handler is async
    """
    def decorator(func):
        event_filter = EventFilter(event_types=event_types)
        event_bus = get_event_bus()
        subscription_id = event_bus.subscribe(func, event_filter, is_async)

        # Store subscription ID on function for later unsubscription
        func._event_subscription_id = subscription_id
        return func

    if handler:
        return decorator(handler)
    return decorator
