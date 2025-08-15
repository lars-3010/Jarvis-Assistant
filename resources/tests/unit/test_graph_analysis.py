"""
Test advanced graph analysis functionality.

This module tests the GraphAnalyzer class and its integration with the dataset
generation pipeline for comprehensive network analysis.
"""

import networkx as nx
import pytest
from unittest.mock import Mock, patch, MagicMock

from jarvis.tools.dataset_generation.analyzers.graph_analyzer import (
    GraphAnalyzer, AdvancedCentralityMetrics, CommunityMetrics, GraphMetrics
)
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.models.data_models import CentralityMetrics
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vault.reader import VaultReader


class TestGraphAnalyzer:
    """Test GraphAnalyzer functionality."""

    @pytest.fixture
    def graph_analyzer(self):
        """Create a graph analyzer instance."""
        return GraphAnalyzer(damping_factor=0.85, max_iter=100, tolerance=1e-6)

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        G = nx.DiGraph()
        G.add_edges_from([
            ('A', 'B'), ('B', 'C'), ('C', 'A'),  # Triangle
            ('A', 'D'), ('D', 'E'),              # Chain
            ('F', 'G')                           # Isolated pair
        ])
        return G

    @pytest.fixture
    def complex_graph(self):
        """Create a more complex test graph with communities."""
        G = nx.DiGraph()
        
        # Community 1: Dense cluster
        community1 = ['A', 'B', 'C', 'D']
        for i, node1 in enumerate(community1):
            for node2 in community1[i+1:]:
                G.add_edge(node1, node2)
                G.add_edge(node2, node1)  # Make it bidirectional
        
        # Community 2: Another dense cluster
        community2 = ['E', 'F', 'G']
        for i, node1 in enumerate(community2):
            for node2 in community2[i+1:]:
                G.add_edge(node1, node2)
                G.add_edge(node2, node1)
        
        # Bridge between communities
        G.add_edge('D', 'E')
        G.add_edge('E', 'D')
        
        # Isolated nodes
        G.add_edge('H', 'I')
        
        return G

    def test_graph_analyzer_initialization(self):
        """Test GraphAnalyzer initialization."""
        analyzer = GraphAnalyzer(damping_factor=0.9, max_iter=500, tolerance=1e-5)
        assert analyzer.damping_factor == 0.9
        assert analyzer.max_iter == 500
        assert analyzer.tolerance == 1e-5

    def test_analyze_empty_graph(self, graph_analyzer):
        """Test analysis of empty graph."""
        empty_graph = nx.DiGraph()
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(empty_graph)
        
        assert node_metrics == {}
        assert community_metrics.num_communities == 0
        assert graph_metrics.num_nodes == 0
        assert graph_metrics.num_edges == 0

    def test_analyze_simple_graph(self, graph_analyzer, simple_graph):
        """Test analysis of simple graph."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(simple_graph)
        
        # Check that we have metrics for all nodes
        assert len(node_metrics) == simple_graph.number_of_nodes()
        for node in simple_graph.nodes():
            assert node in node_metrics
            metrics = node_metrics[node]
            assert isinstance(metrics, AdvancedCentralityMetrics)
            
            # Check that basic metrics are computed
            assert 0.0 <= metrics.degree_centrality <= 1.0
            assert metrics.pagerank > 0.0  # PageRank should be positive
            assert metrics.clustering_coefficient >= 0.0
        
        # Check graph-level metrics
        assert graph_metrics.num_nodes == simple_graph.number_of_nodes()
        assert graph_metrics.num_edges == simple_graph.number_of_edges()
        assert 0.0 <= graph_metrics.density <= 1.0
        
        # Check community detection
        assert community_metrics.num_communities > 0
        assert -1.0 <= community_metrics.modularity <= 1.0

    def test_pagerank_calculation(self, graph_analyzer, simple_graph):
        """Test PageRank calculation."""
        node_metrics, _, _ = graph_analyzer.analyze_graph(simple_graph)
        
        # PageRank values should sum to approximately 1.0
        pagerank_sum = sum(metrics.pagerank for metrics in node_metrics.values())
        assert abs(pagerank_sum - 1.0) < 0.01
        
        # All PageRank values should be positive
        for metrics in node_metrics.values():
            assert metrics.pagerank > 0.0

    def test_structural_holes_metrics(self, graph_analyzer, complex_graph):
        """Test structural holes and brokerage metrics."""
        node_metrics, _, _ = graph_analyzer.analyze_graph(complex_graph)
        
        # Check that constraint and effective size are calculated
        for node, metrics in node_metrics.items():
            assert 0.0 <= metrics.constraint <= 1.0
            assert metrics.effective_size >= 0.0

    def test_community_detection(self, graph_analyzer, complex_graph):
        """Test community detection functionality."""
        _, community_metrics, _ = graph_analyzer.analyze_graph(complex_graph)
        
        # Should detect communities
        assert community_metrics.num_communities > 1
        assert len(community_metrics.communities) == community_metrics.num_communities
        
        # All nodes should be assigned to communities
        total_nodes_in_communities = sum(len(nodes) for nodes in community_metrics.communities.values())
        assert total_nodes_in_communities == complex_graph.number_of_nodes()

    def test_top_nodes_by_metric(self, graph_analyzer, complex_graph):
        """Test getting top nodes by specific metrics."""
        node_metrics, _, _ = graph_analyzer.analyze_graph(complex_graph)
        
        # Get top nodes by PageRank
        top_pagerank = graph_analyzer.get_top_nodes_by_metric(node_metrics, 'pagerank', top_k=3)
        
        assert len(top_pagerank) <= 3
        assert len(top_pagerank) <= len(node_metrics)
        
        # Check that results are sorted in descending order
        for i in range(len(top_pagerank) - 1):
            assert top_pagerank[i][1] >= top_pagerank[i+1][1]


class TestGraphAnalysisIntegration:
    """Test graph analysis integration with dataset generation."""

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock vector encoder."""
        encoder = Mock(spec=VectorEncoder)
        encoder.vector_dim = 384
        encoder.encode_batch.return_value = [[0.1] * 384] * 10
        return encoder

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock vault reader."""
        reader = Mock(spec=VaultReader)
        reader.read_notes.return_value = [
            {
                'path': 'note1.md',
                'content': 'This is note 1 content',
                'metadata': {'title': 'Note 1'}
            },
            {
                'path': 'note2.md', 
                'content': 'This is note 2 content',
                'metadata': {'title': 'Note 2'}
            }
        ]
        return reader

    def test_advanced_graph_metrics_integration(self, mock_vector_encoder, mock_vault_reader):
        """Test integration of advanced graph metrics in dataset generation."""
        # Create generator with mocked dependencies
        generator = NotesDatasetGenerator(
            vector_encoder=mock_vector_encoder,
            vault_reader=mock_vault_reader,
            use_tfidf=True,
            use_content_analysis=True
        )
        
        # Create a simple graph for testing
        test_graph = nx.DiGraph()
        test_graph.add_edges_from([
            ('note1.md', 'note2.md'),
            ('note2.md', 'note3.md'),
            ('note3.md', 'note1.md')
        ])
        
        # Test centrality computation
        centrality_metrics = generator._precompute_centrality_metrics(test_graph)
        
        # Should have metrics for all nodes
        assert len(centrality_metrics) == test_graph.number_of_nodes()
        
        # Check that advanced metrics are included
        for node, metrics in centrality_metrics.items():
            assert isinstance(metrics, CentralityMetrics)
            
            # Basic centrality metrics
            assert hasattr(metrics, 'degree_centrality')
            assert hasattr(metrics, 'betweenness_centrality')
            assert hasattr(metrics, 'closeness_centrality')
            assert hasattr(metrics, 'eigenvector_centrality')
            
            # Advanced centrality metrics
            assert hasattr(metrics, 'pagerank')
            assert hasattr(metrics, 'katz_centrality')
            assert hasattr(metrics, 'harmonic_centrality')
            
            # Structural metrics
            assert hasattr(metrics, 'clustering_coefficient')
            assert hasattr(metrics, 'constraint')
            assert hasattr(metrics, 'effective_size')
            
            # Community metrics
            assert hasattr(metrics, 'community_id')
            assert hasattr(metrics, 'modularity_contribution')
            
            # Path metrics
            assert hasattr(metrics, 'eccentricity')
            assert hasattr(metrics, 'average_shortest_path')
            
            # Check value ranges
            assert 0.0 <= metrics.degree_centrality <= 1.0
            assert metrics.pagerank > 0.0
            assert metrics.clustering_coefficient >= 0.0
            assert 0.0 <= metrics.constraint <= 1.0
            assert metrics.effective_size >= 0.0

    def test_fallback_centrality_calculation(self, mock_vector_encoder, mock_vault_reader):
        """Test fallback to basic centrality when advanced analysis fails."""
        generator = NotesDatasetGenerator(
            vector_encoder=mock_vector_encoder,
            vault_reader=mock_vault_reader
        )
        
        # Mock the graph analyzer to raise an exception
        with patch.object(generator.graph_analyzer, 'analyze_graph', side_effect=Exception("Analysis failed")):
            test_graph = nx.DiGraph()
            test_graph.add_edges_from([('A', 'B'), ('B', 'C')])
            
            # Should fall back to basic centrality
            centrality_metrics = generator._precompute_centrality_metrics(test_graph)
            
            # Should still have metrics for all nodes
            assert len(centrality_metrics) == test_graph.number_of_nodes()
            
            # Should have basic degree centrality
            for metrics in centrality_metrics.values():
                assert metrics.degree_centrality >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])