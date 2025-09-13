"""
GraphRAG Service Implementation (features).
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

            result = await self._synthesize_result(
                query_config, ranked_results, graph_data, clusters, start_time
            )

            execution_time_ms = int((time.time() - start_time) * 1000)
            if self.metrics:
                self.metrics.record_histogram(
                    "graphrag_search_duration_ms", execution_time_ms, tags={"mode": query_config.mode}
                )
                self.metrics.record_histogram(
                    "graphrag_search_results", len(result.sources), tags={"mode": query_config.mode}
                )

            logger.info(
                f"GraphRAG search completed in {execution_time_ms}ms, found {len(result.sources)} sources"
            )
            return result

        except Exception as e:
            logger.error(f"GraphRAG search failed: {e}")
            if self.metrics:
                self.metrics.record_counter("graphrag_search_errors")
            raise ServiceError(f"GraphRAG search failed: {e}") from e

    async def _semantic_retrieval_phase(self, query_config: GraphRAGQuery) -> list[SearchResult]:
        if query_config.mode == "quick":
            top_k = min(query_config.max_sources * 2, 15)
        elif query_config.mode == "focused":
            top_k = min(query_config.max_sources * 3, 25)
        else:
            top_k = min(query_config.max_sources * 4, 40)

        logger.debug(
            f"Semantic retrieval: top_k={top_k}, vault={query_config.vault_name}"
        )

        semantic_results = self.retriever.retrieve(
            query=query_config.query, top_k=top_k, vault_name=query_config.vault_name
        )

        if query_config.semantic_threshold is not None:
            semantic_results = [
                r
                for r in semantic_results
                if r.similarity_score is not None
                and r.similarity_score >= query_config.semantic_threshold
            ]

        logger.debug(f"Found {len(semantic_results)} semantic results")
        return semantic_results

    async def _graph_expansion_phase(
        self, semantic_results: list[SearchResult], query_config: GraphRAGQuery
    ) -> dict[str, Any]:
        graph_data = {
            "neighborhoods": {},
            "all_nodes": set(),
            "all_relationships": [],
            "connectivity_scores": {},
            "expansion_stats": {
                "centers_explored": 0,
                "nodes_discovered": 0,
                "relationships_discovered": 0,
            },
        }

        if query_config.mode == "quick":
            centers_to_explore = semantic_results[: query_config.max_sources]
        elif query_config.mode == "focused":
            centers_to_explore = semantic_results[: query_config.max_sources + 2]
        else:
            centers_to_explore = semantic_results[: query_config.max_sources * 2]

        logger.debug(
            f"Graph expansion: exploring {len(centers_to_explore)} centers with depth {query_config.depth}"
        )

        for result in centers_to_explore:
            try:
                center_path = str(result.path)
                neighborhood = self.graph_fetcher.fetch(
                    center_path, depth=query_config.depth
                )

                if neighborhood and neighborhood.get("nodes"):
                    graph_data["neighborhoods"][center_path] = neighborhood
                    nodes = neighborhood.get("nodes", [])
                    relationships = neighborhood.get("relationships", [])
                    graph_data["all_nodes"].update(
                        n.get("path", "") for n in nodes if isinstance(n, dict)
                    )
                    graph_data["all_relationships"].extend(relationships)
                    graph_data["connectivity_scores"][center_path] = len(relationships) + 0.5 * len(nodes)
                    graph_data["expansion_stats"]["centers_explored"] += 1
                    graph_data["expansion_stats"]["nodes_discovered"] += len(nodes)
                    graph_data["expansion_stats"]["relationships_discovered"] += len(relationships)
            except Exception as e:
                logger.debug(f"Graph expansion failed for {result.path}: {e}")

        logger.debug(f"Graph expansion complete: {graph_data['expansion_stats']}")
        return graph_data

    async def _enhanced_reranking_phase(
        self,
        semantic_results: list[SearchResult],
        graph_data: dict[str, Any],
        query_config: GraphRAGQuery,
    ) -> list[tuple[SearchResult, float, dict[str, Any]]]:
        graphs_for_reranker = [
            (str(result.path), graph_data["neighborhoods"].get(str(result.path), {}))
            for result in semantic_results
        ]

        basic_ranked = self.reranker.rerank(semantic_results, graphs_for_reranker)

        enhanced_ranked = []
        for result, basic_score in basic_ranked:
            path = str(result.path)
            connectivity_score = graph_data["connectivity_scores"].get(path, 0.0)
            neighborhood_size = len(
                graph_data["neighborhoods"].get(path, {}).get("nodes", [])
            )

            centrality_boost = min(1.0, connectivity_score / 20.0) * 0.2
            neighborhood = graph_data["neighborhoods"].get(path, {})
            rel_types = set()
            for rel in neighborhood.get("relationships", []):
                if isinstance(rel, dict):
                    rel_types.add(rel.get("original_type", rel.get("type", "UNKNOWN")))
            diversity_boost = min(1.0, len(rel_types) / 5.0) * 0.1

            enhanced_score = (
                basic_score * 0.7
                + centrality_boost
                + diversity_boost
                + (query_config.graph_boost * min(1.0, connectivity_score / 10.0))
            )

            metadata = {
                "basic_score": basic_score,
                "connectivity_score": connectivity_score,
                "neighborhood_size": neighborhood_size,
                "centrality_boost": centrality_boost,
                "diversity_boost": diversity_boost,
                "relationship_types": list(rel_types),
            }

            enhanced_ranked.append((result, enhanced_score, metadata))

        enhanced_ranked.sort(key=lambda x: x[1], reverse=True)
        return enhanced_ranked

    async def _content_enrichment_phase(
        self,
        ranked_results: list[tuple[SearchResult, float, dict[str, Any]]],
        query_config: GraphRAGQuery,
    ) -> None:
        if not self.vault_reader:
            return

        for i, (result, _, metadata) in enumerate(ranked_results[:5]):
            try:
                preview = self.vault_reader.read_note_preview(str(result.path), 200)
                metadata["content_preview"] = preview
                metadata["content_length"] = len(preview) if preview else 0
            except Exception:
                pass

    async def _cluster_analysis_phase(
        self,
        ranked_results: list[tuple[SearchResult, float, dict[str, Any]]],
        graph_data: dict[str, Any],
        query_config: GraphRAGQuery,
    ) -> list[GraphRAGCluster]:
        clusters: list[GraphRAGCluster] = []
        if len(ranked_results) < 5:
            return clusters

        try:
            paths = [str(result.path) for result, _, _ in ranked_results]
            connected_components = self._find_connected_components(paths, graph_data)
            for i, component in enumerate(connected_components):
                if len(component) >= 2:
                    relationship_types = []
                    for path in component:
                        neighborhood = graph_data["neighborhoods"].get(path, {})
                        for rel in neighborhood.get("relationships", []):
                            if isinstance(rel, dict):
                                rel_type = rel.get("original_type", rel.get("type", ""))
                                if rel_type:
                                    relationship_types.append(rel_type)

                    theme = (
                        max(set(relationship_types), key=relationship_types.count)
                        if relationship_types
                        else f"Cluster {i+1}"
                    )

                    cluster = GraphRAGCluster(
                        cluster_id=f"cluster_{i+1}",
                        theme=theme,
                        confidence=min(1.0, len(component) / len(paths)),
                        node_paths=component,
                        keywords=list(set(relationship_types[:5])),
                    )
                    clusters.append(cluster)

            logger.debug(f"Cluster analysis found {len(clusters)} clusters")
        except Exception as e:
            logger.warning(f"Cluster analysis failed: {e}")

        return clusters

    def _find_connected_components(self, paths: list[str], graph_data: dict[str, Any]) -> list[list[str]]:
        adjacency = {path: set() for path in paths}
        for path in paths:
            neighborhood = graph_data["neighborhoods"].get(path, {})
            for rel in neighborhood.get("relationships", []):
                if isinstance(rel, dict):
                    source = rel.get("source_path", "")
                    target = rel.get("target_path", "")
                    if source in adjacency and target in adjacency:
                        adjacency[source].add(target)
                        adjacency[target].add(source)

        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for path in paths:
            if path not in visited:
                component: list[str] = []
                dfs(path, component)
                components.append(component)

        return components

    async def _synthesize_result(
        self,
        query_config: GraphRAGQuery,
        ranked_results: list[tuple[SearchResult, float, dict[str, Any]]],
        graph_data: dict[str, Any],
        clusters: list[GraphRAGCluster],
        start_time: float,
    ) -> GraphRAGResult:
        execution_time_ms = int((time.time() - start_time) * 1000)

        sources = []
        for i, (result, enhanced_score, metadata) in enumerate(ranked_results):
            source = {
                "rank": i + 1,
                "path": str(result.path),
                "vault_name": result.vault_name,
                "semantic_score": float(result.similarity_score)
                if result.similarity_score
                else 0.0,
                "enhanced_score": float(enhanced_score),
                "connectivity_score": metadata.get("connectivity_score", 0.0),
                "neighborhood_size": metadata.get("neighborhood_size", 0),
                "relationship_types": metadata.get("relationship_types", []),
            }
            if "content_preview" in metadata:
                source["content_preview"] = metadata["content_preview"]
                source["content_length"] = metadata.get("content_length", 0)
            sources.append(source)

        nodes = []
        for path in graph_data["all_nodes"]:
            connectivity_score = graph_data["connectivity_scores"].get(path, 0.0)
            relationship_count = 0
            for rel in graph_data["all_relationships"]:
                if isinstance(rel, dict):
                    if rel.get("source_path") == path or rel.get("target_path") == path:
                        relationship_count += 1

            node = GraphRAGNode(
                path=path,
                vault_name=query_config.vault_name or "unknown",
                centrality_score=connectivity_score,
                relationship_count=relationship_count,
            )
            nodes.append(node)

        relationships = []
        processed_rels = set()
        for rel in graph_data["all_relationships"]:
            if isinstance(rel, dict):
                source = rel.get("source_path", "")
                target = rel.get("target_path", "")
                rel_type = rel.get("original_type", rel.get("type", "UNKNOWN"))
                rel_key = f"{source}->{target}:{rel_type}"
                if rel_key not in processed_rels:
                    relationship = GraphRAGRelationship(
                        source_path=source,
                        target_path=target,
                        relationship_type=rel_type,
                        strength=1.0,
                        context=rel.get("context"),
                    )
                    relationships.append(relationship)
                    processed_rels.add(rel_key)

        graph_metrics = {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "connected_components": len(clusters),
            "average_connectivity": sum(graph_data["connectivity_scores"].values())
            / max(1, len(graph_data["connectivity_scores"])),
            "expansion_coverage": graph_data["expansion_stats"]["centers_explored"]
            / max(1, query_config.max_sources),
        }

        search_path = [str(result.path) for result, _, _ in ranked_results[:5]]

        result = GraphRAGResult(
            query=query_config.query,
            execution_time_ms=execution_time_ms,
            sources=sources,
            nodes=nodes,
            relationships=relationships,
            clusters=clusters,
            metadata={
                "mode": query_config.mode,
                "max_sources": query_config.max_sources,
                "depth": query_config.depth,
                "vault_name": query_config.vault_name,
                "include_content": query_config.include_content,
                "semantic_threshold": query_config.semantic_threshold,
                "graph_boost": query_config.graph_boost,
            },
            graph_metrics=graph_metrics,
            search_path=search_path,
        )

        return result

    def _empty_result(self, query_config: GraphRAGQuery, start_time: float) -> GraphRAGResult:
        return GraphRAGResult(
            query=query_config.query,
            execution_time_ms=int((time.time() - start_time) * 1000),
            sources=[],
            nodes=[],
            relationships=[],
            clusters=[],
            metadata=asdict(query_config),
            graph_metrics={
                "total_nodes": 0,
                "total_relationships": 0,
                "connected_components": 0,
                "average_connectivity": 0.0,
                "expansion_coverage": 0.0,
            },
            search_path=[],
        )
