from jarvis.core.interfaces import IGraphDatabase


class GraphNeighborhoodFetcher:
    """Fetches bounded neighborhoods around a center note."""

    def __init__(self, graph_db: IGraphDatabase):
        self.graph_db = graph_db

    def fetch(self, center_path: str, depth: int = 1) -> dict:
        return self.graph_db.get_note_graph(center_path, depth)

