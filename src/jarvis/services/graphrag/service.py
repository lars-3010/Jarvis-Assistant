"""
GraphRAG Service Implementation (services canonical).
"""

import time
from dataclasses import asdict, dataclass
from typing import Any

from jarvis.core.interfaces import IGraphDatabase, IMetrics, IVaultReader, IVectorSearcher
from jarvis.models.document import SearchResult
from jarvis.services.search.ranking import ResultRanker
from jarvis.utils.errors import ServiceError
import logging

from .graph_fetcher import GraphNeighborhoodFetcher
from .reranker import GraphRAGReranker
from .retriever import GraphRAGRetriever

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGQuery:
    query: str
    mode: str = "quick"
    max_sources: int = 5
    depth: int = 1
    vault_name: str | None = None
    include_content: bool = False
    semantic_threshold: float | None = None
    graph_boost: float = 0.3


@dataclass
class GraphRAGNode:
    path: str
    vault_name: str
    title: str | None = None
    content_preview: str | None = None
    node_type: str = "note"
    centrality_score: float = 0.0
    relationship_count: int = 0


@dataclass
class GraphRAGRelationship:
    source_path: str
    target_path: str
    relationship_type: str
    strength: float = 1.0
    context: str | None = None


@dataclass
class GraphRAGCluster:
    cluster_id: str
    theme: str
    confidence: float
    node_paths: list[str]
    keywords: list[str]


@dataclass
class GraphRAGResult:
    query: str
    execution_time_ms: int
    sources: list[dict[str, Any]]
    nodes: list[GraphRAGNode]
    relationships: list[GraphRAGRelationship]
    clusters: list[GraphRAGCluster]
    metadata: dict[str, Any]
    graph_metrics: dict[str, Any]
    search_path: list[str]


class GraphRAGService:
    def __init__(
        self,
        vector_searcher: IVectorSearcher,
        graph_database: IGraphDatabase,
        vault_reader: IVaultReader | None = None,
        metrics: IMetrics | None = None,
        result_ranker: ResultRanker | None = None,
    ):
        self.vector_searcher = vector_searcher
        self.graph_database = graph_database
        self.vault_reader = vault_reader
        self.metrics = metrics
        self.result_ranker = result_ranker

        self.retriever = GraphRAGRetriever(vector_searcher)
        self.graph_fetcher = GraphNeighborhoodFetcher(graph_database)
        self.reranker = GraphRAGReranker()

        logger.info("GraphRAG service initialized")

    async def search(self, query_config: GraphRAGQuery) -> GraphRAGResult:
        start_time = time.time()

        try:
            if self.metrics:
                self.metrics.record_counter(
                    "graphrag_search_requests", tags={"mode": query_config.mode}
                )

            semantic_results = await self._semantic_retrieval_phase(query_config)
            if not semantic_results:
                logger.warning("No semantic results found")
                return self._empty_result(query_config, start_time)

            graph_data = await self._graph_expansion_phase(semantic_results, query_config)
            ranked_results = await self._enhanced_reranking_phase(
                semantic_results, graph_data, query_config
            )

            if query_config.include_content:
                await self._content_enrichment_phase(ranked_results, query_config)

            clusters = await self._cluster_analysis_phase(
                ranked_results, graph_data, query_config
            )

            execution_time_ms = int((time.time() - start_time) * 1000)
            result = GraphRAGResult(
                query=query_config.query,
                execution_time_ms=execution_time_ms,
                sources=[asdict(sr) if hasattr(sr, "path") else sr for sr in semantic_results],
                nodes=self._build_nodes(ranked_results),
                relationships=self._build_relationships(graph_data),
                clusters=clusters,
                metadata={
                    "mode": query_config.mode,
                    "max_sources": query_config.max_sources,
                    "depth": query_config.depth,
                },
                graph_metrics=self._compute_graph_metrics(graph_data),
                search_path=["semantic", "graph_expansion", "rerank", "clusters"],
            )

            if self.metrics:
                self.metrics.record_histogram("graphrag_search_latency_ms", execution_time_ms)

            return result

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            if self.metrics:
                self.metrics.record_counter("graphrag_search_errors")
            raise ServiceError(f"GraphRAG search failed: {e}")

    async def _semantic_retrieval_phase(self, query_config: GraphRAGQuery) -> list[SearchResult]:
        results = self.retriever.retrieve(
            query=query_config.query,
            top_k=query_config.max_sources,
            vault_name=query_config.vault_name,
        )

        if query_config.semantic_threshold is not None:
            results = [r for r in results if (r.similarity_score or 0.0) >= query_config.semantic_threshold]
        return results

    async def _graph_expansion_phase(self, semantic_results: list[SearchResult], query_config: GraphRAGQuery) -> list[tuple[str, dict]]:
        graphs: list[tuple[str, dict]] = []
        for sr in semantic_results:
            center = str(sr.path)
            g = self.graph_fetcher.fetch(center, depth=query_config.depth)
            graphs.append((center, g))
        return graphs

    async def _enhanced_reranking_phase(
        self,
        semantic_results: list[SearchResult],
        graphs: list[tuple[str, dict]],
        query_config: GraphRAGQuery,
    ) -> list[tuple[SearchResult, float]]:
        reranked = self.reranker.rerank(semantic_results, graphs)
        if self.result_ranker:
            # Optionally merge with keyword/other evidence via ResultRanker
            reranked = reranked  # placeholder for future integration
        return reranked

    async def _content_enrichment_phase(
        self,
        ranked_results: list[tuple[SearchResult, float]],
        query_config: GraphRAGQuery,
    ) -> None:
        # Placeholder for optional content loading enrichment
        return None

    async def _cluster_analysis_phase(
        self,
        ranked_results: list[tuple[SearchResult, float]],
        graphs: list[tuple[str, dict]],
        query_config: GraphRAGQuery,
    ) -> list[GraphRAGCluster]:
        # Placeholder for simple clustering; can use tags or folders as proxies
        return []

    def _build_nodes(self, ranked: list[tuple[SearchResult, float]]) -> list[GraphRAGNode]:
        nodes: list[GraphRAGNode] = []
        for sr, score in ranked:
            nodes.append(
                GraphRAGNode(
                    path=str(sr.path),
                    vault_name=sr.vault_name,
                    title=None,
                    content_preview=None,
                    centrality_score=score,
                    relationship_count=0,
                )
            )
        return nodes

    def _build_relationships(self, graphs: list[tuple[str, dict]]) -> list[GraphRAGRelationship]:
        # Extract relationships from graph neighborhood data
        rels: list[GraphRAGRelationship] = []
        for center, g in graphs:
            for r in g.get("relationships", []):
                rels.append(
                    GraphRAGRelationship(
                        source_path=r.get("source", center),
                        target_path=r.get("target", ""),
                        relationship_type=r.get("type", "link"),
                        strength=float(r.get("weight", 1.0)),
                        context=r.get("context"),
                    )
                )
        return rels

    def _compute_graph_metrics(self, graphs: list[tuple[str, dict]]) -> dict[str, Any]:
        # Simple metrics summarization
        total_nodes = 0
        total_rels = 0
        for _, g in graphs:
            total_nodes += len(g.get("nodes", []))
            total_rels += len(g.get("relationships", []))
        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
        }

    def _empty_result(self, query_config: GraphRAGQuery, start_time: float) -> GraphRAGResult:
        return GraphRAGResult(
            query=query_config.query,
            execution_time_ms=int((time.time() - start_time) * 1000),
            sources=[],
            nodes=[],
            relationships=[],
            clusters=[],
            metadata={
                "mode": query_config.mode,
                "max_sources": query_config.max_sources,
                "depth": query_config.depth,
            },
            graph_metrics={"total_nodes": 0, "total_relationships": 0},
            search_path=["semantic"],
        )

__all__ = [
    "GraphRAGQuery",
    "GraphRAGNode",
    "GraphRAGRelationship",
    "GraphRAGCluster",
    "GraphRAGResult",
    "GraphRAGService",
]
