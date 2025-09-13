"""
Lightweight test container helper to provide services by interface.

Usage:
    from resources.tests.helpers.container import make_container
    container = make_container(searcher=mock_searcher, vault_reader=mock_vault_reader)
    plugin = SomePlugin(container=container)
"""

from __future__ import annotations

from typing import Any


class TestContainer:
    def __init__(self, mapping: dict[type, Any] | None = None):
        self._mapping = mapping or {}

    def register(self, iface: type, impl: Any) -> None:
        self._mapping[iface] = impl

    def get(self, iface: type) -> Any:
        return self._mapping.get(iface)


def make_container(
    *,
    searcher: Any | None = None,
    vault_reader: Any | None = None,
    graph_db: Any | None = None,
    health_checker: Any | None = None,
) -> TestContainer:
    """Create a test container, mapping common interfaces if provided."""
    from jarvis.core.interfaces import (
        IGraphDatabase,
        IHealthChecker,
        IVaultReader,
        IVectorSearcher,
    )

    mapping: dict[type, Any] = {}
    if searcher is not None:
        mapping[IVectorSearcher] = searcher
    if vault_reader is not None:
        mapping[IVaultReader] = vault_reader
    if graph_db is not None:
        mapping[IGraphDatabase] = graph_db
    if health_checker is not None:
        mapping[IHealthChecker] = health_checker

    return TestContainer(mapping)

