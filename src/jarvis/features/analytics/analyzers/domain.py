"""
Knowledge domain analyzer for semantic clustering and relationship mapping.
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from jarvis.core.interfaces import IGraphDatabase, IVaultReader, IVectorSearcher

from ..errors import (
    AnalysisTimeoutError,
    InsufficientDataError,
    ModelError,
    ServiceUnavailableError,
)
from ..models import (
    AnalyticsError,
    BridgeOpportunity,
    DomainConnection,
    KnowledgeDomain,
    SemanticCluster,
)

logger = logging.getLogger(__name__)


class KnowledgeDomainAnalyzer:
    """Analyzes knowledge domains and their relationships within the vault."""

    def __init__(
        self,
        vector_searcher: IVectorSearcher | None = None,
        graph_db: IGraphDatabase | None = None,
        vault_reader: IVaultReader | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.vector_searcher = vector_searcher
        self.graph_db = graph_db
        self.vault_reader = vault_reader
        self.config = config or {}

        self.max_processing_time = self.config.get("max_processing_time_seconds", 15)
        self.clustering_threshold = self.config.get("clustering_threshold", 0.7)
        self.min_cluster_size = self.config.get("min_cluster_size", 3)
        self.max_domains = self.config.get("max_domains", 20)

        self._note_embeddings: dict[str, np.ndarray] = {}
        self._note_metadata: dict[str, dict[str, Any]] = {}
        self._similarity_cache: dict[tuple[str, str], float] = {}

        logger.debug("KnowledgeDomainAnalyzer initialized")

    async def cluster_by_semantic_similarity(
        self, embeddings: dict[str, np.ndarray]
    ) -> list[SemanticCluster]:
        start_time = time.time()

        try:
            if len(embeddings) < self.min_cluster_size:
                raise InsufficientDataError(
                    "domain_analyzer", self.min_cluster_size, len(embeddings)
                )

            note_paths = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[path] for path in note_paths])

            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            normalized_embeddings = embedding_matrix / np.maximum(norms, 1e-8)

            clusters = await self._perform_clustering(
                normalized_embeddings, note_paths
            )

            semantic_clusters = []
            for i, (cluster_indices, coherence) in enumerate(clusters):
                if len(cluster_indices) < self.min_cluster_size:
                    continue

                cluster_notes = [note_paths[idx] for idx in cluster_indices]
                centroid_idx = self._find_centroid_note(
                    cluster_indices, normalized_embeddings
                )
                centroid_note = note_paths[centroid_idx]

                keywords = await self._extract_cluster_keywords(cluster_notes)

                description = self._generate_cluster_description(
                    cluster_notes, keywords
                )

                semantic_cluster = SemanticCluster(
                    id=f"cluster_{i}",
                    centroid_note=centroid_note,
                    notes=cluster_notes,
                    coherence_score=coherence,
                    keywords=keywords,
                    description=description,
                )
                semantic_clusters.append(semantic_cluster)

                if time.time() - start_time > self.max_processing_time:
                    logger.warning("Semantic clustering timeout")
                    break

            semantic_clusters.sort(key=lambda c: c.coherence_score, reverse=True)
            return semantic_clusters[: self.max_domains]

        except Exception as e:
            if isinstance(e, (InsufficientDataError, AnalysisTimeoutError)):
                raise
            logger.error(f"Semantic clustering error: {e}")
            raise ModelError("semantic_clustering", "clustering", str(e))

    async def analyze_cross_domain_connections(
        self, graph_data: dict[str, Any]
    ) -> list[DomainConnection]:
        try:
            if not self.graph_db or not self.graph_db.is_healthy:
                raise ServiceUnavailableError(
                    "graph_database", "domain connection analysis"
                )

            domain_connections = []
            clusters = await self._get_existing_clusters(graph_data)
            domain_map = self._map_notes_to_domains(clusters)

            connection_counts = defaultdict(lambda: defaultdict(int))
            bridge_notes = defaultdict(lambda: defaultdict(list))

            for note_path, connections in graph_data.get("connections", {}).items():
                source_domain = domain_map.get(note_path)
                if not source_domain:
                    continue

                for connected_note in connections.get("outbound", []):
                    target_domain = domain_map.get(connected_note)
                    if target_domain and target_domain != source_domain:
                        connection_counts[source_domain][target_domain] += 1
                        bridge_notes[source_domain][target_domain].append(note_path)

            for from_domain, targets in connection_counts.items():
                for to_domain, count in targets.items():
                    if count >= 2:
                        strength = self._calculate_connection_strength(
                            from_domain, to_domain, count, domain_map
                        )
                        connection_type = self._classify_connection_type(
                            from_domain, to_domain, bridge_notes[from_domain][to_domain]
                        )

                        domain_connection = DomainConnection(
                            from_domain=from_domain,
                            to_domain=to_domain,
                            connection_strength=strength,
                            connection_count=count,
                            bridge_notes=bridge_notes[from_domain][to_domain],
                            connection_type=connection_type,
                        )
                        domain_connections.append(domain_connection)

            logger.debug(
                f"Domain connection analysis found {len(domain_connections)} connections"
            )
            return domain_connections

        except Exception as e:
            if isinstance(e, ServiceUnavailableError):
                raise
            logger.error(f"Domain connection analysis error: {e}")
            raise AnalyticsError(
                f"Failed to analyze domain connections: {e}",
                "domain_analyzer",
                "connection_analysis",
            )

    # --- internal helper stubs (omitted; to be implemented incrementally) ---
    async def _perform_clustering(self, normalized_embeddings, note_paths):
        return []

    def _find_centroid_note(self, cluster_indices, normalized_embeddings) -> int:
        return cluster_indices[0] if cluster_indices else 0

    async def _extract_cluster_keywords(self, cluster_notes: list[str]) -> list[str]:
        return []

    def _generate_cluster_description(self, cluster_notes: list[str], keywords: list[str]) -> str:
        return ", ".join(keywords) or "Thematic cluster"

    async def _get_existing_clusters(self, graph_data: dict[str, Any]) -> list[SemanticCluster]:
        return []

    def _map_notes_to_domains(self, clusters: list[SemanticCluster]) -> dict[str, str]:
        return {}

    def _calculate_connection_strength(
        self, from_domain: str, to_domain: str, count: int, domain_map: dict[str, str]
    ) -> float:
        return min(1.0, count / 10.0)

    def _classify_connection_type(
        self, from_domain: str, to_domain: str, notes: list[str]
    ) -> str:
        return "associative"

