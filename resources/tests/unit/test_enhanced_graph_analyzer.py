"""
Unit tests for enhanced GraphAnalyzer with bridge nodes and temporal patterns.
"""

import pytest
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from jarvis.tools.dataset_generation.analyzers.graph_analyzer import (
    GraphAnalyzer,
    AdvancedCentralityMetrics,
    CommunityMetrics,
    GraphMetrics
)


class TestEnhancedGraphAnalyzer:
    """Test enhanced graph analytics features."""
    
    @pytest.fixture
    def graph_analyzer(self):
        """Create a GraphAnalyzer instance."""
        return GraphAnalyzer()
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample directed graph for testing."""
        graph = nx.DiGraph()
        
        # Add nodes with some attributes
        nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for node in nodes:
            graph.add_node(node)
        
        # Add edges to create communities and bridges
        # Community 1: A, B, C
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        
        # Community 2: E, F, G
        graph.add_edges_from([('E', 'F'), ('F', 'G'), ('G', 'E')])
        
        # Bridge connections through D and H
        graph.add_edges_from([('D', 'A'), ('D', 'E')])  # D bridges communities
        graph.add_edges_from([('H', 'C'), ('H', 'F')])  # H also bridges
        
        return graph
    
    @pytest.fixture
    def temporal_graph(self):
        """Create a graph with temporal information."""
        graph = nx.DiGraph()
        
        # Create temporary files to simulate real file paths
        temp_dir = tempfile.mkdtemp()
        
        # Add nodes with file paths and timestamps
        base_time = datetime.now().timestamp()
        nodes_data = [
            ('note1.md', base_time - 86400),  # 1 day ago
            ('note2.md', base_time - 3600),   # 1 hour ago
            ('note3.md', base_time - 1800),   # 30 minutes ago
            ('note4.md', base_time),          # now
        ]
        
        for i, (filename, timestamp) in enumerate(nodes_data):
            filepath = os.path.join(temp_dir, filename)
            # Create actual files with timestamps
            with open(filepath, 'w') as f:
                f.write(f"Content of {filename}")
            
            # Set file modification time
            os.utime(filepath, (timestamp, timestamp))
            
            # Add node with temporal attributes
            graph.add_node(filepath, creation_time=timestamp, modified_time=timestamp)
        
        # Add some edges
        filepaths = [os.path.join(temp_dir, filename) for filename, _ in nodes_data]
        graph.add_edges_from([
            (filepaths[0], filepaths[1]),
            (filepaths[1], filepaths[2]),
            (filepaths[2], filepaths[3]),
            (filepaths[3], filepaths[0])
        ])
        
        return graph, temp_dir
    
    def test_eigenvector_centrality_computation(self, graph_analyzer, sample_graph):
        """Test that eigenvector centrality is computed correctly."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Check that eigenvector centrality is computed for all nodes
        for node, metrics in node_metrics.items():
            assert hasattr(metrics, 'eigenvector_centrality')
            assert isinstance(metrics.eigenvector_centrality, float)
            assert 0.0 <= metrics.eigenvector_centrality <= 1.0
        
        # Nodes in well-connected components should have higher eigenvector centrality
        assert node_metrics['A'].eigenvector_centrality > 0
        assert node_metrics['B'].eigenvector_centrality > 0
        assert node_metrics['C'].eigenvector_centrality > 0
    
    def test_community_detection_algorithms(self, graph_analyzer, sample_graph):
        """Test that community detection works with fallback algorithms."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Check that communities were detected
        assert community_metrics.num_communities > 0
        assert len(community_metrics.communities) > 0
        assert len(community_metrics.node_communities) > 0
        
        # Check that modularity is reasonable
        assert -1.0 <= community_metrics.modularity <= 1.0
        
        # Check that all nodes are assigned to communities
        for node in sample_graph.nodes():
            assert node in community_metrics.node_communities
            community_id = community_metrics.node_communities[node]
            assert community_id in community_metrics.communities
            assert node in community_metrics.communities[community_id]
    
    def test_bridge_node_identification(self, graph_analyzer, sample_graph):
        """Test bridge node identification functionality."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Check that bridge metrics are computed
        for node, metrics in node_metrics.items():
            assert hasattr(metrics, 'is_bridge_node')
            assert hasattr(metrics, 'bridge_score')
            assert hasattr(metrics, 'inter_community_connections')
            
            assert isinstance(metrics.is_bridge_node, bool)
            assert isinstance(metrics.bridge_score, float)
            assert isinstance(metrics.inter_community_connections, int)
            
            assert 0.0 <= metrics.bridge_score <= 1.0
            assert metrics.inter_community_connections >= 0
        
        # D and H should be identified as bridge nodes (they connect different communities)
        bridge_nodes = [node for node, metrics in node_metrics.items() if metrics.is_bridge_node]
        
        # We should have some bridge nodes
        assert len(bridge_nodes) > 0
        
        # Bridge nodes should have higher bridge scores
        non_bridge_nodes = [node for node, metrics in node_metrics.items() if not metrics.is_bridge_node]
        if bridge_nodes and non_bridge_nodes:
            avg_bridge_score = sum(node_metrics[node].bridge_score for node in bridge_nodes) / len(bridge_nodes)
            avg_non_bridge_score = sum(node_metrics[node].bridge_score for node in non_bridge_nodes) / len(non_bridge_nodes)
            assert avg_bridge_score > avg_non_bridge_score
    
    def test_temporal_pattern_analysis(self, graph_analyzer, temporal_graph):
        """Test temporal pattern analysis functionality."""
        temporal_graph_obj, temp_dir = temporal_graph
        
        try:
            node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(temporal_graph_obj)
            
            # Check that temporal metrics are computed
            nodes_with_temporal_data = 0
            for node, metrics in node_metrics.items():
                assert hasattr(metrics, 'creation_time')
                assert hasattr(metrics, 'modification_time')
                assert hasattr(metrics, 'temporal_activity_score')
                
                if metrics.creation_time is not None or metrics.modification_time is not None:
                    nodes_with_temporal_data += 1
                    
                    # Temporal activity score should be between 0 and 1
                    assert 0.0 <= metrics.temporal_activity_score <= 1.0
            
            # We should have temporal data for all nodes
            assert nodes_with_temporal_data > 0
            
            # More recent nodes should have higher activity scores
            temporal_nodes = [(node, metrics) for node, metrics in node_metrics.items() 
                            if metrics.modification_time is not None]
            
            if len(temporal_nodes) > 1:
                # Sort by modification time
                temporal_nodes.sort(key=lambda x: x[1].modification_time or 0)
                
                # More recent nodes should generally have higher activity scores
                recent_node = temporal_nodes[-1][1]  # Most recent
                old_node = temporal_nodes[0][1]      # Oldest
                
                # Recent node should have higher or equal activity score
                assert recent_node.temporal_activity_score >= old_node.temporal_activity_score
        
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_get_bridge_nodes_method(self, graph_analyzer, sample_graph):
        """Test the get_bridge_nodes method."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Get top bridge nodes
        bridge_nodes = graph_analyzer.get_bridge_nodes(node_metrics, top_k=5)
        
        # Should return a list of tuples
        assert isinstance(bridge_nodes, list)
        
        if bridge_nodes:
            for node_name, bridge_score in bridge_nodes:
                assert isinstance(node_name, str)
                assert isinstance(bridge_score, float)
                assert 0.0 <= bridge_score <= 1.0
                
                # Verify this node is actually marked as a bridge node
                assert node_metrics[node_name].is_bridge_node
            
            # Should be sorted by bridge score (descending)
            scores = [score for _, score in bridge_nodes]
            assert scores == sorted(scores, reverse=True)
    
    def test_get_temporal_insights_method(self, graph_analyzer, temporal_graph):
        """Test the get_temporal_insights method."""
        temporal_graph_obj, temp_dir = temporal_graph
        
        try:
            node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(temporal_graph_obj)
            
            # Get temporal insights
            insights = graph_analyzer.get_temporal_insights(node_metrics)
            
            # Check expected keys
            expected_keys = [
                'nodes_with_temporal_data',
                'average_activity_score',
                'most_active_nodes',
                'temporal_coverage',
                'creation_time_range',
                'modification_time_range'
            ]
            
            for key in expected_keys:
                assert key in insights
            
            # Check data types and ranges
            assert isinstance(insights['nodes_with_temporal_data'], int)
            assert insights['nodes_with_temporal_data'] >= 0
            
            assert isinstance(insights['average_activity_score'], float)
            assert 0.0 <= insights['average_activity_score'] <= 1.0
            
            assert isinstance(insights['most_active_nodes'], list)
            assert isinstance(insights['temporal_coverage'], float)
            assert 0.0 <= insights['temporal_coverage'] <= 1.0
            
            # If we have temporal data, ranges should be tuples
            if insights['nodes_with_temporal_data'] > 0:
                if insights['creation_time_range']:
                    assert isinstance(insights['creation_time_range'], tuple)
                    assert len(insights['creation_time_range']) == 2
                
                if insights['modification_time_range']:
                    assert isinstance(insights['modification_time_range'], tuple)
                    assert len(insights['modification_time_range']) == 2
        
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_graceful_degradation_empty_graph(self, graph_analyzer):
        """Test that analyzer handles empty graphs gracefully."""
        empty_graph = nx.DiGraph()
        
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(empty_graph)
        
        # Should return empty results without errors
        assert node_metrics == {}
        assert community_metrics.num_communities == 0
        assert graph_metrics.num_nodes == 0
        assert graph_metrics.num_edges == 0
    
    def test_graceful_degradation_single_node(self, graph_analyzer):
        """Test that analyzer handles single-node graphs gracefully."""
        single_node_graph = nx.DiGraph()
        single_node_graph.add_node('A')
        
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(single_node_graph)
        
        # Should handle single node without errors
        assert len(node_metrics) == 1
        assert 'A' in node_metrics
        
        # Metrics should have reasonable default values
        metrics = node_metrics['A']
        # For a single node, degree centrality is 1.0 in NetworkX (normalized)
        assert metrics.degree_centrality >= 0.0  # Should be non-negative
        assert metrics.eigenvector_centrality >= 0.0
        assert not metrics.is_bridge_node  # Can't be a bridge with no connections
    
    def test_community_detection_fallback(self, graph_analyzer, sample_graph):
        """Test that community detection falls back gracefully when advanced algorithms fail."""
        # This test verifies that the fallback mechanism works
        # The actual fallback is tested by the algorithm trying different methods
        
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Should always detect some communities (even if just using greedy modularity)
        assert community_metrics.num_communities > 0
        assert len(community_metrics.communities) > 0
        
        # All nodes should be assigned to communities
        assert len(community_metrics.node_communities) == sample_graph.number_of_nodes()
    
    def test_advanced_centrality_metrics(self, graph_analyzer, sample_graph):
        """Test that all advanced centrality metrics are computed."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Check that all expected centrality metrics are present
        expected_metrics = [
            'degree_centrality',
            'closeness_centrality', 
            'eigenvector_centrality',
            'betweenness_centrality',
            'pagerank',
            'katz_centrality',
            'harmonic_centrality'
        ]
        
        for node, metrics in node_metrics.items():
            for metric_name in expected_metrics:
                assert hasattr(metrics, metric_name)
                value = getattr(metrics, metric_name)
                assert isinstance(value, float)
                assert value >= 0.0  # All centrality measures should be non-negative
    
    def test_structural_metrics(self, graph_analyzer, sample_graph):
        """Test structural metrics computation."""
        node_metrics, community_metrics, graph_metrics = graph_analyzer.analyze_graph(sample_graph)
        
        # Check structural metrics
        for node, metrics in node_metrics.items():
            assert hasattr(metrics, 'clustering_coefficient')
            assert hasattr(metrics, 'constraint')
            assert hasattr(metrics, 'effective_size')
            
            assert 0.0 <= metrics.clustering_coefficient <= 1.0
            assert 0.0 <= metrics.constraint <= 1.0
            assert metrics.effective_size >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])