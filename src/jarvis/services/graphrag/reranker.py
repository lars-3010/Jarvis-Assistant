from jarvis.models.document import SearchResult


class GraphRAGReranker:
    """Simple heuristic reranker combining semantic and graph features."""

    def rerank(
        self,
        semantic_results: list[SearchResult],
        graphs: list[tuple[str, dict]],
    ) -> list[tuple[SearchResult, float]]:
        graph_scores = {}
        for center, g in graphs:
            nodes = g.get("nodes", []) if isinstance(g, dict) else []
            rels = g.get("relationships", []) if isinstance(g, dict) else []
            graph_scores[center] = 0.5 * len(rels) + 0.2 * len(nodes)

        ranked: list[tuple[SearchResult, float]] = []
        for res in semantic_results:
            path = str(res.path)
            sem = max(0.0, 1.0 - max(0.0, float(res.similarity_score))) if res.similarity_score is not None else 0.0
            gscore = graph_scores.get(path, 0.0)
            unified = 0.7 * sem + 0.3 * min(1.0, gscore / 10.0)
            ranked.append((res, unified))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

__all__ = ["GraphRAGReranker"]
