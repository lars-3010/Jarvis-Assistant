import threading
import time

import pytest

from jarvis.core.container import ServiceContainer
from jarvis.core.events import (
    MemoryEventStore,
    get_event_bus,
    publish_event_threadsafe,
)
from jarvis.utils.config import JarvisSettings


class _DummyService:
    def __init__(self, value: int = 0):
        self.value = value


def _dummy_factory():
    return _DummyService(value=42)


def test_register_factory_smoke():
    settings = JarvisSettings()
    container = ServiceContainer(settings)

    # Register a factory for an interface type (using the concrete class as key)
    container.register_factory(_DummyService, _dummy_factory, singleton=True)

    # Resolve via interface key and verify singleton behavior
    svc1 = container.get(_DummyService)
    svc2 = container.get(_DummyService)

    assert isinstance(svc1, _DummyService)
    assert svc1.value == 42
    assert svc1 is svc2  # singleton


@pytest.mark.anyio
async def test_publish_event_threadsafe_smoke():
    # Use an in-memory event store and set global bus to use it
    store = MemoryEventStore()
    bus = get_event_bus(event_store=store)
    await bus.start()

    # Publish from another thread
    def _publish():
        ok = publish_event_threadsafe(
            event_type="unit.test",
            data={"k": "v"},
            source="test_thread",
        )
        assert ok is True

    t = threading.Thread(target=_publish)
    t.start()
    t.join(timeout=2.0)

    # Allow event loop to process
    await asyncio_sleep(0.1)

    events = await bus.get_events(event_types={"unit.test"})
    assert len(events) >= 1
    assert events[-1].data.get("k") == "v"

    await bus.stop()


# Compatibility wrapper to avoid importing asyncio in test header
async def asyncio_sleep(delay: float):
    import anyio
    await anyio.sleep(delay)
