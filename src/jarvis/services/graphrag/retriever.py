from jarvis.core.interfaces import IVectorSearcher
from jarvis.models.document import SearchResult


class GraphRAGRetriever:
    """Semantic retriever for GraphRAG MVP."""

    def __init__(self, searcher: IVectorSearcher):
        self.searcher = searcher

    def retrieve(self, query: str, top_k: int = 10, vault_name: str | None = None) -> list[SearchResult]:
        return self.searcher.search(query=query, top_k=top_k, vault_name=vault_name)

__all__ = ["GraphRAGRetriever"]
