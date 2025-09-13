"""
Vault analytics orchestrator service (features).

Coordinates analyzers to provide a unified interface for vault analysis.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from jarvis.core.event_integration import EventTypes
from jarvis.core.events import (
    Event,
    EventFilter,
    get_event_bus,
    publish_event_threadsafe,
)
from jarvis.core.interfaces import (
    IGraphDatabase,
    IMetrics,
    IVaultAnalyticsService,
    IVaultReader,
    IVectorSearcher,
)
from jarvis.utils.config import get_settings

from .analyzers.domain import KnowledgeDomainAnalyzer
from .analyzers.quality import ContentQualityAnalyzer
from .analyzers.structure import VaultStructureAnalyzer
from .cache import AnalyticsCache
from .errors import (
    AnalysisTimeoutError,
    InsufficientDataError,
    ServiceUnavailableError,
    VaultNotFoundError,
)
from .models import (
    ActionableRecommendation,
    AnalyticsError,
    BridgeOpportunity,
    DomainMap,
    FolderHierarchy,
    KnowledgeDomain,
    OrganizationMethod,
    OrganizationPattern,
    QualityAnalysis,
    QualityGap,
    QualityScore,
    VaultContext,
    DepthMetrics,
)


logger = logging.getLogger(__name__)


class VaultAnalyticsService(IVaultAnalyticsService):
    """
    Main analytics service that orchestrates vault analysis.
    """

    def __init__(
        self,
        vault_reader: Optional[IVaultReader] = None,
        vector_searcher: Optional[IVectorSearcher] = None,
        graph_db: Optional[IGraphDatabase] = None,
        metrics: Optional[IMetrics] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vault_reader = vault_reader
        self.vector_searcher = vector_searcher
        self.graph_db = graph_db
        self.metrics = metrics

        settings = get_settings()
        self.config = config or settings.get_analytics_config()
        self.enabled = self.config.get("enabled", True)

        if not self.enabled:
            logger.info("Analytics service disabled by configuration")
            return

        # Initialize cache and analyzers
        self.cache = AnalyticsCache(self.config.get("cache", {}))
        self.structure_analyzer = VaultStructureAnalyzer(
            config=self.config.get("performance", {})
        )
        self.quality_analyzer = ContentQualityAnalyzer(
            vault_reader=vault_reader, graph_db=graph_db, config=self.config.get("quality", {})
        )
        self.domain_analyzer = KnowledgeDomainAnalyzer(
            vector_searcher=vector_searcher,
            graph_db=graph_db,
            vault_reader=vault_reader,
            config=self.config.get("domains", {}),
        )

        self.max_processing_time = self.config.get("performance", {}).get(
            "max_processing_time_seconds", 15
        )
        self.enable_parallel_processing = self.config.get("performance", {}).get(
            "enable_parallel_processing", True
        )
        self.sample_large_vaults = self.config.get("performance", {}).get(
            "sample_large_vaults", True
        )
        self.sample_threshold = self.config.get("performance", {}).get(
            "sample_threshold", 5000
        )

        logger.info("VaultAnalyticsService initialized")

        # Subscribe to vault-related events for cache invalidation
        try:
            self._event_bus = get_event_bus()
            self._subscription_ids: list[str] = []

            vault_filter = EventFilter(
                event_types={
                    EventTypes.VAULT_INDEXED,
                    EventTypes.VAULT_UPDATED,
                    EventTypes.DOCUMENT_ADDED,
                    EventTypes.DOCUMENT_UPDATED,
                    EventTypes.DOCUMENT_DELETED,
                }
            )

            sub_id = self._event_bus.subscribe(
                self._handle_vault_event, vault_filter, is_async=True
            )
            self._subscription_ids.append(sub_id)
            logger.info("Analytics service subscribed to vault events for cache invalidation")
        except Exception as e:
            logger.warning(f"Analytics event subscription failed: {e}")

    async def get_vault_context(self, vault_name: str = "default") -> Dict[str, Any]:
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "vault context generation")

        start = time.time()

        # Basic stats may be cached
        cache_key = f"context:{vault_name}"
        cached = self.cache.get_quick(cache_key)
        if cached:
            return cached

        # Gather files and structure
        files = await self._get_vault_files(vault_name)

        # Structure analysis
        org_pattern = await self.structure_analyzer.detect_organization_method(files)
        depth_metrics = await self.structure_analyzer.calculate_depth_metrics(files)
        folder_structure = await self._build_folder_hierarchy(files)

        # Quality distribution and domains
        quality_dist, avg_quality = await self._analyze_quality_distribution(vault_name)
        quality_trends: list = []
        domains, domain_connections, isolated_notes = await self._map_domains(vault_name)

        # Recommendations and gaps (placeholders for now)
        recommendations: list[ActionableRecommendation] = []
        quality_gaps: list[QualityGap] = []
        bridge_opportunities: list[BridgeOpportunity] = []

        processing_time_ms = int((time.time() - start) * 1000)

        context = VaultContext(
            total_notes=len(files),
            total_size_bytes=0,
            last_updated=time.time(),
            analysis_timestamp=time.time(),
            vault_name=vault_name,
            organization_pattern=org_pattern,
            folder_structure=folder_structure,
            depth_metrics=depth_metrics,
            quality_distribution=quality_dist,
            average_quality_score=avg_quality,
            quality_trends=quality_trends,
            identified_domains=domains,
            domain_connections=domain_connections,
            isolated_notes=isolated_notes,
            processing_time_ms=processing_time_ms,
            cache_hit_rate=0.0,
            confidence_score=0.8,
            recommendations=recommendations,
            quality_gaps=quality_gaps,
            bridge_opportunities=bridge_opportunities,
            cache_status=self.cache.get_status(),
        )

        self.cache.put_quick(cache_key, asdict(context))
        return asdict(context)

    async def analyze_quality_distribution(self, vault_name: str = "default") -> Dict[str, Any]:
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "quality distribution analysis")
        # Placeholder: reuse get_vault_context for now
        context = await self.get_vault_context(vault_name)
        return {
            "average_quality": context.get("average_quality_score", 0.0),
            "quality_distribution": context.get("quality_distribution", {}),
            "processing_time_ms": context.get("processing_time_ms", 0),
        }

    async def map_knowledge_domains(self, vault_name: str = "default") -> Dict[str, Any]:
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "domain mapping")
        domains, connections, isolated = await self._map_domains(vault_name)
        return {
            "domains": [asdict(d) for d in domains],
            "domain_connections": [asdict(c) for c in connections],
            "isolated_notes": isolated,
        }

    async def assess_note_quality(self, note_path: str, vault_name: str = "default") -> Dict[str, Any]:
        if not self.enabled:
            raise ServiceUnavailableError("analytics_service", "note quality assessment")
        if not self.vault_reader:
            raise ServiceUnavailableError("vault_reader", "note quality assessment")

        content, metadata = self.vault_reader.read_note_with_metadata(note_path)
        score: QualityScore = await self.quality_analyzer.assess_note_quality(
            content, metadata, note_path
        )
        return asdict(score)

    async def get_analytics_cache_status(self) -> Dict[str, Any]:
        """Return cache status with backward-compatible keys for plugins."""
        status = self.cache.get_status()
        data = asdict(status)

        # Back-compat field aliases expected by some tools
        data.setdefault("total_size_mb", data.get("memory_usage_mb", 0.0))
        # Human-friendly timestamp
        try:
            import datetime as _dt
            ts = float(data.get("last_cleanup", 0.0))
            data.setdefault("last_updated_str", _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            data.setdefault("last_updated_str", "Unknown")

        # Level breakdown
        data.setdefault("levels", {
            "1": {"entry_count": data.get("l1_entries", 0)},
            "2": {"entry_count": data.get("l2_entries", 0)},
            "3": {"entry_count": data.get("l3_entries", 0)},
        })

        # Freshness indicators
        data.setdefault("freshness_indicators", {
            "oldest_entry_age_minutes": data.get("oldest_entry_age_minutes", 0.0)
        })

        return data

    async def invalidate_cache(self, vault_name: str | None = None) -> bool:
        self.cache.clear()
        # Broadcast cache-cleared event so other components can react
        try:
            publish_event_threadsafe(
                EventTypes.CACHE_CLEARED,
                {
                    "cache_type": "analytics_cache",
                    "vault_name": vault_name or "*",
                    "reason": "explicit_invalidation",
                },
                source="analytics_service",
            )
        except Exception:
            pass
        return True

    async def get_recommendations(self, vault_name: str = "default", limit: int = 10) -> Dict[str, Any]:
        return {"recommendations": []}

    # --- internal helpers ---
    async def _get_vault_files(self, vault_name: str) -> list[Path]:
        if not self.vault_reader:
            raise ServiceUnavailableError("vault_reader", "vault analysis")

        files = self.vault_reader.list_notes(vault_name)
        if not files:
            raise VaultNotFoundError(vault_name)
        # Sample large vaults if configured
        if self.sample_large_vaults and len(files) > self.sample_threshold:
            return files[: self.sample_threshold]
        return files

    async def _build_folder_hierarchy(self, files: list[Path]) -> FolderHierarchy:
        # Quick heuristic; full implementation can be refined later
        max_depth = max((len(p.parts) - 1 for p in files), default=0)
        total_folders = len({str(p.parent) for p in files})
        avg_depth = sum(len(p.parts) - 1 for p in files) / max(1, len(files))
        roots = sorted({p.parts[0] for p in files if p.parts})
        return FolderHierarchy(
            max_depth=max_depth,
            average_depth=avg_depth,
            total_folders=total_folders,
            root_folders=roots,
            deepest_paths=sorted({str(p) for p in files}, key=lambda s: len(s.split("/")), reverse=True)[:5],
            empty_folders=[],
        )

    async def _analyze_quality_distribution(self, vault_name: str) -> tuple[dict[str, int], float]:
        # For now, a placeholder distribution
        dist = {"🌱": 1, "🌿": 1, "🌳": 1, "🗺️": 1}
        avg = 0.5
        return dist, avg

    async def _map_domains(
        self, vault_name: str
    ) -> tuple[list[KnowledgeDomain], list, list[str]]:
        # Placeholder domain mapping
        return [], [], []

    def _handle_vault_event(self, event: Event) -> None:
        try:
            # Invalidate caches on vault-related events
            self.cache.clear()
            # Notify the system that analytics cache was cleared due to a vault change
            try:
                publish_event_threadsafe(
                    EventTypes.CACHE_CLEARED,
                    {
                        "cache_type": "analytics_cache",
                        "vault_name": event.data.get("vault_name", "unknown"),
                        "path": event.data.get("path"),
                        "reason": event.data.get("operation", "vault_event"),
                    },
                    source="analytics_service",
                )
            except Exception:
                pass
        except Exception:
            pass
