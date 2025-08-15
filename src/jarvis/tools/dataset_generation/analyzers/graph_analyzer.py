"""
Advanced graph analyzer for comprehensive network analysis.

This module provides advanced graph analysis including PageRank centrality,
community detection, structural holes analysis, and path-based metrics.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import datetime
from pathlib import Path

from jarvis.utils.logging import setup_logging
from ..error_handling import (
    with_error_handling, ComponentType, ErrorSeverity, FallbackValues,
    get_error_tracker
)

logger = setup_logging(__name__)


@dataclass
class AdvancedCentralityMetrics:
    """Advanced centrality and graph metrics for a node."""
    # Basic centrality metrics
    degree_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    
    # Advanced centrality metrics
    pagerank: float = 0.0
    katz_centrality: float = 0.0
    harmonic_centrality: float = 0.0
    
    # Structural metrics
    clustering_coefficient: float = 0.0
    constraint: float = 0.0  # Structural holes metric
    effective_size: float = 0.0  # Brokerage metric
    
    # Community metrics
    community_id: int = -1
    modularity_contribution: float = 0.0
    
    # Bridge metrics
    is_bridge_node: bool = False
    bridge_score: float = 0.0  # How important this node is as a bridge
    inter_community_connections: int = 0  # Number of connections to other communities
    
    # Path-based metrics
    eccentricity: float = 0.0
    average_shortest_path: float = 0.0
    
    # Temporal metrics
    creation_time: Optional[float] = None
    modification_time: Optional[float] = None
    temporal_activity_score: float = 0.0  # Activity level over time


@dataclass
class CommunityMetrics:
    """Community detection results."""
    communities: Dict[int, Set[str]] = field(default_factory=dict)  # community_id -> nodes
    node_communities: Dict[str, int] = field(default_factory=dict)  # node -> community_id
    modularity: float = 0.0
    num_communities: int = 0
    community_sizes: Dict[int, int] = field(default_factory=dict)


@dataclass
class GraphMetrics:
    """Overall graph-level metrics."""
    # Basic graph properties
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    
    # Connectivity metrics
    num_components: int = 0
    largest_component_size: int = 0
    average_clustering: float = 0.0
    
    # Path metrics
    diameter: float = 0.0
    average_path_length: float = 0.0
    
    # Community metrics
    modularity: float = 0.0
    num_communities: int = 0
    
    # Centralization metrics
    degree_centralization: float = 0.0
    betweenness_centralization: float = 0.0


class GraphAnalyzer:
    """Advanced graph analysis for knowledge networks."""
    
    def __init__(self, damping_factor: float = 0.85, max_iter: int = 1000, tolerance: float = 1e-6):
        """Initialize the graph analyzer.
        
        Args:
            damping_factor: PageRank damping factor
            max_iter: Maximum iterations for iterative algorithms
            tolerance: Convergence tolerance
        """
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def analyze_graph(self, graph: nx.DiGraph) -> Tuple[Dict[str, AdvancedCentralityMetrics], CommunityMetrics, GraphMetrics]:
        """Perform comprehensive graph analysis with robust error handling.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Tuple of (node_metrics, community_metrics, graph_metrics)
        """
        if not graph or graph.number_of_nodes() == 0:
            return {}, CommunityMetrics(), GraphMetrics()
        
        logger.info(f"Analyzing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Calculate node-level metrics with error handling
        node_metrics = self._calculate_node_metrics_safe(graph)
        
        # Detect communities with error handling
        community_metrics = self._detect_communities_safe(graph)
        
        # Calculate graph-level metrics with error handling
        graph_metrics = self._calculate_graph_metrics_safe(graph, community_metrics)
        
        # Add community information to node metrics with error handling
        self._add_community_info_to_nodes_safe(node_metrics, community_metrics)
        
        # Identify bridge nodes with error handling
        self._identify_bridge_nodes_safe(graph, node_metrics, community_metrics)
        
        # Analyze temporal patterns if timestamps are available with error handling
        self._analyze_temporal_patterns_safe(graph, node_metrics)
        
        return node_metrics, community_metrics, graph_metrics
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="node_metrics_calculation",
        fallback_value={},
        severity=ErrorSeverity.HIGH,
        recovery_action="Check if NetworkX is properly installed and graph is valid"
    )
    def _calculate_node_metrics_safe(self, graph: nx.DiGraph) -> Dict[str, AdvancedCentralityMetrics]:
        """Safely calculate node metrics with error handling."""
        return self._calculate_node_metrics(graph)
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="community_detection",
        fallback_value=CommunityMetrics(),
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Community detection will be skipped - check if community detection libraries are available"
    )
    def _detect_communities_safe(self, graph: nx.DiGraph) -> CommunityMetrics:
        """Safely detect communities with error handling."""
        return self._detect_communities(graph)
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="graph_metrics_calculation",
        fallback_value=GraphMetrics(),
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Graph-level metrics will use default values"
    )
    def _calculate_graph_metrics_safe(self, graph: nx.DiGraph, community_metrics: CommunityMetrics) -> GraphMetrics:
        """Safely calculate graph metrics with error handling."""
        return self._calculate_graph_metrics(graph, community_metrics)
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="community_info_addition",
        fallback_value=None,
        severity=ErrorSeverity.LOW,
        recovery_action="Community information will not be added to node metrics"
    )
    def _add_community_info_to_nodes_safe(self, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                                        community_metrics: CommunityMetrics):
        """Safely add community info to nodes with error handling."""
        self._add_community_info_to_nodes(node_metrics, community_metrics)
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="bridge_node_identification",
        fallback_value=None,
        severity=ErrorSeverity.LOW,
        recovery_action="Bridge nodes will not be identified"
    )
    def _identify_bridge_nodes_safe(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                                  community_metrics: CommunityMetrics):
        """Safely identify bridge nodes with error handling."""
        self._identify_bridge_nodes(graph, node_metrics, community_metrics)
    
    @with_error_handling(
        component=ComponentType.GRAPH_ANALYZER,
        feature_name="temporal_pattern_analysis",
        fallback_value=None,
        severity=ErrorSeverity.LOW,
        recovery_action="Temporal patterns will not be analyzed"
    )
    def _analyze_temporal_patterns_safe(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Safely analyze temporal patterns with error handling."""
        self._analyze_temporal_patterns(graph, node_metrics)
    
    def _calculate_node_metrics(self, graph: nx.DiGraph) -> Dict[str, AdvancedCentralityMetrics]:
        """Calculate comprehensive node-level metrics.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping node names to AdvancedCentralityMetrics
        """
        node_metrics = {}
        
        try:
            # Initialize metrics for all nodes
            for node in graph.nodes():
                node_metrics[node] = AdvancedCentralityMetrics()
            
            # Basic centrality metrics
            self._calculate_basic_centrality(graph, node_metrics)
            
            # Advanced centrality metrics
            self._calculate_advanced_centrality(graph, node_metrics)
            
            # Structural metrics
            self._calculate_structural_metrics(graph, node_metrics)
            
            # Path-based metrics
            self._calculate_path_metrics(graph, node_metrics)
            
        except Exception as e:
            logger.error(f"Error calculating node metrics: {e}")
        
        return node_metrics
    
    def _calculate_basic_centrality(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Calculate basic centrality metrics."""
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            for node, value in degree_centrality.items():
                node_metrics[node].degree_centrality = value
            
            # Betweenness centrality (computationally expensive, so we sample for large graphs)
            if graph.number_of_nodes() > 1000:
                # Sample nodes for large graphs
                k = min(100, graph.number_of_nodes() // 10)
                betweenness = nx.betweenness_centrality(graph, k=k, normalized=True)
            else:
                betweenness = nx.betweenness_centrality(graph, normalized=True)
            
            for node, value in betweenness.items():
                node_metrics[node].betweenness_centrality = value
            
            # Closeness and eigenvector centrality for connected components
            for component in nx.weakly_connected_components(graph):
                if len(component) > 1:
                    subgraph = graph.subgraph(component)
                    
                    # Closeness centrality
                    try:
                        closeness = nx.closeness_centrality(subgraph)
                        for node, value in closeness.items():
                            node_metrics[node].closeness_centrality = value
                    except Exception as e:
                        logger.warning(f"Failed to calculate closeness centrality: {e}")
                    
                    # Eigenvector centrality
                    try:
                        eigenvector = nx.eigenvector_centrality(subgraph, max_iter=self.max_iter)
                        for node, value in eigenvector.items():
                            node_metrics[node].eigenvector_centrality = value
                    except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
                        logger.warning(f"Failed to calculate eigenvector centrality: {e}")
                        # Use degree centrality as fallback
                        for node in component:
                            node_metrics[node].eigenvector_centrality = node_metrics[node].degree_centrality
        
        except Exception as e:
            logger.error(f"Error calculating basic centrality: {e}")
    
    def _calculate_advanced_centrality(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Calculate advanced centrality metrics."""
        try:
            # PageRank
            pagerank = nx.pagerank(graph, alpha=self.damping_factor, max_iter=self.max_iter, tol=self.tolerance)
            for node, value in pagerank.items():
                node_metrics[node].pagerank = value
            
            # Katz centrality (for connected components)
            for component in nx.weakly_connected_components(graph):
                if len(component) > 1:
                    subgraph = graph.subgraph(component)
                    try:
                        # Use a small alpha to ensure convergence
                        alpha = 0.1 / max(dict(subgraph.degree()).values()) if subgraph.degree() else 0.01
                        katz = nx.katz_centrality(subgraph, alpha=alpha, max_iter=self.max_iter, tol=self.tolerance)
                        for node, value in katz.items():
                            node_metrics[node].katz_centrality = value
                    except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
                        logger.warning(f"Failed to calculate Katz centrality: {e}")
                        # Use PageRank as fallback
                        for node in component:
                            node_metrics[node].katz_centrality = node_metrics[node].pagerank
            
            # Harmonic centrality
            try:
                harmonic = nx.harmonic_centrality(graph)
                for node, value in harmonic.items():
                    node_metrics[node].harmonic_centrality = float(value)
            except Exception as e:
                logger.warning(f"Failed to calculate harmonic centrality: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating advanced centrality: {e}")
    
    def _calculate_structural_metrics(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Calculate structural metrics including clustering and structural holes."""
        try:
            # Clustering coefficient
            clustering = nx.clustering(graph.to_undirected())
            for node, value in clustering.items():
                node_metrics[node].clustering_coefficient = value
            
            # Structural holes metrics (constraint and effective size)
            undirected_graph = graph.to_undirected()
            
            for node in graph.nodes():
                try:
                    # Calculate constraint (Burt's structural holes measure)
                    constraint = self._calculate_constraint(undirected_graph, node)
                    node_metrics[node].constraint = constraint
                    
                    # Calculate effective size (brokerage measure)
                    effective_size = self._calculate_effective_size(undirected_graph, node)
                    node_metrics[node].effective_size = effective_size
                    
                except Exception as e:
                    logger.debug(f"Failed to calculate structural metrics for node {node}: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating structural metrics: {e}")
    
    def _calculate_constraint(self, graph: nx.Graph, node: str) -> float:
        """Calculate Burt's constraint measure for structural holes.
        
        Args:
            graph: Undirected graph
            node: Node to calculate constraint for
            
        Returns:
            Constraint value (0-1, lower values indicate more structural holes)
        """
        if node not in graph or graph.degree(node) == 0:
            return 1.0  # Maximum constraint for isolated nodes
        
        neighbors = list(graph.neighbors(node))
        if len(neighbors) <= 1:
            return 1.0
        
        constraint = 0.0
        total_ties = graph.degree(node)
        
        for neighbor in neighbors:
            # Direct tie strength (proportion of ego's ties)
            direct_tie = 1.0 / total_ties
            
            # Indirect ties through mutual neighbors
            indirect_ties = 0.0
            mutual_neighbors = set(graph.neighbors(neighbor)) & set(neighbors)
            mutual_neighbors.discard(node)  # Remove ego from mutual neighbors
            
            for mutual in mutual_neighbors:
                if graph.has_edge(node, mutual):
                    indirect_ties += (1.0 / total_ties) * (1.0 / graph.degree(mutual))
            
            # Total constraint from this neighbor
            neighbor_constraint = (direct_tie + indirect_ties) ** 2
            constraint += neighbor_constraint
        
        return min(1.0, constraint)
    
    def _calculate_effective_size(self, graph: nx.Graph, node: str) -> float:
        """Calculate effective size (brokerage measure).
        
        Args:
            graph: Undirected graph
            node: Node to calculate effective size for
            
        Returns:
            Effective size (number of non-redundant contacts)
        """
        if node not in graph:
            return 0.0
        
        neighbors = list(graph.neighbors(node))
        if len(neighbors) <= 1:
            return len(neighbors)
        
        # Count redundant ties (ties between neighbors)
        redundant_ties = 0.0
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i+1:]:
                if graph.has_edge(neighbor1, neighbor2):
                    # Each mutual connection reduces effective size
                    redundant_ties += 1.0 / len(neighbors)
        
        effective_size = len(neighbors) - redundant_ties
        return max(0.0, effective_size)
    
    def _calculate_path_metrics(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Calculate path-based metrics."""
        try:
            # For large graphs, sample components to avoid computational explosion
            for component in nx.weakly_connected_components(graph):
                if len(component) > 500:  # Skip very large components
                    logger.warning(f"Skipping path metrics for large component with {len(component)} nodes")
                    continue
                
                if len(component) > 1:
                    subgraph = graph.subgraph(component)
                    
                    try:
                        # Calculate shortest path lengths
                        path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
                        
                        for node in component:
                            if node in path_lengths:
                                lengths = list(path_lengths[node].values())
                                if lengths:
                                    # Eccentricity (maximum shortest path from this node)
                                    node_metrics[node].eccentricity = max(lengths)
                                    
                                    # Average shortest path length
                                    node_metrics[node].average_shortest_path = sum(lengths) / len(lengths)
                    
                    except Exception as e:
                        logger.warning(f"Failed to calculate path metrics for component: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating path metrics: {e}")
    
    def _detect_communities(self, graph: nx.DiGraph) -> CommunityMetrics:
        """Detect communities using advanced algorithms (Leiden/Louvain).
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            CommunityMetrics object with community detection results
        """
        community_metrics = CommunityMetrics()
        
        if graph.number_of_nodes() == 0:
            return community_metrics
        
        try:
            # Convert to undirected for community detection
            undirected_graph = graph.to_undirected()
            
            # Try Leiden algorithm first (best quality)
            partition, modularity = self._try_leiden_detection(undirected_graph)
            
            # If Leiden fails, try Louvain algorithm
            if partition is None:
                partition, modularity = self._try_louvain_detection(undirected_graph)
            
            # If both fail, use NetworkX greedy modularity
            if partition is None:
                partition, modularity = self._try_greedy_modularity_detection(undirected_graph)
            
            # Process community results
            if partition is not None:
                community_metrics.modularity = modularity
                community_metrics.node_communities = partition
                
                # Group nodes by community
                for node, community_id in partition.items():
                    if community_id not in community_metrics.communities:
                        community_metrics.communities[community_id] = set()
                    community_metrics.communities[community_id].add(node)
                
                # Calculate community sizes
                for community_id, nodes in community_metrics.communities.items():
                    community_metrics.community_sizes[community_id] = len(nodes)
                
                community_metrics.num_communities = len(community_metrics.communities)
                
                logger.info(f"Detected {community_metrics.num_communities} communities with modularity {modularity:.3f}")
            else:
                logger.warning("All community detection methods failed")
        
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
        
        return community_metrics
    
    def _try_leiden_detection(self, graph: nx.Graph) -> Tuple[Optional[Dict[str, int]], float]:
        """Try Leiden algorithm for community detection.
        
        Args:
            graph: Undirected NetworkX graph
            
        Returns:
            Tuple of (partition dict, modularity) or (None, 0.0) if failed
        """
        try:
            # Try to import leidenalg (requires igraph)
            import igraph as ig
            import leidenalg
            
            # Convert NetworkX graph to igraph
            # Create mapping from node names to indices
            node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
            idx_to_node = {idx: node for node, idx in node_to_idx.items()}
            
            # Create edge list with indices
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
            
            # Create igraph graph
            ig_graph = ig.Graph(n=len(graph.nodes()), edges=edges, directed=False)
            
            # Run Leiden algorithm
            partition_ig = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            
            # Convert back to NetworkX node names
            partition = {}
            for idx, community_id in enumerate(partition_ig.membership):
                node = idx_to_node[idx]
                partition[node] = community_id
            
            modularity = partition_ig.modularity
            
            logger.info("Successfully used Leiden algorithm for community detection")
            return partition, modularity
        
        except ImportError:
            logger.debug("leidenalg or igraph not available, trying Louvain")
            return None, 0.0
        except Exception as e:
            logger.warning(f"Leiden algorithm failed: {e}")
            return None, 0.0
    
    def _try_louvain_detection(self, graph: nx.Graph) -> Tuple[Optional[Dict[str, int]], float]:
        """Try Louvain algorithm for community detection.
        
        Args:
            graph: Undirected NetworkX graph
            
        Returns:
            Tuple of (partition dict, modularity) or (None, 0.0) if failed
        """
        try:
            # Try to import python-louvain
            import community as community_louvain
            
            partition = community_louvain.best_partition(graph)
            modularity = community_louvain.modularity(partition, graph)
            
            logger.info("Successfully used Louvain algorithm for community detection")
            return partition, modularity
        
        except ImportError:
            logger.debug("python-louvain not available, trying greedy modularity")
            return None, 0.0
        except Exception as e:
            logger.warning(f"Louvain algorithm failed: {e}")
            return None, 0.0
    
    def _try_greedy_modularity_detection(self, graph: nx.Graph) -> Tuple[Optional[Dict[str, int]], float]:
        """Try NetworkX greedy modularity for community detection.
        
        Args:
            graph: Undirected NetworkX graph
            
        Returns:
            Tuple of (partition dict, modularity) or (None, 0.0) if failed
        """
        try:
            communities_generator = nx.community.greedy_modularity_communities(graph)
            communities_list = list(communities_generator)
            
            # Convert to partition format
            partition = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    partition[node] = i
            
            modularity = nx.community.modularity(graph, communities_list)
            
            logger.info("Successfully used greedy modularity for community detection")
            return partition, modularity
        
        except Exception as e:
            logger.error(f"Greedy modularity detection failed: {e}")
            return None, 0.0
    
    def _add_community_info_to_nodes(self, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                                   community_metrics: CommunityMetrics):
        """Add community information to node metrics."""
        for node, metrics in node_metrics.items():
            if node in community_metrics.node_communities:
                metrics.community_id = community_metrics.node_communities[node]
                
                # Calculate modularity contribution (simplified)
                community_size = community_metrics.community_sizes.get(metrics.community_id, 1)
                metrics.modularity_contribution = 1.0 / community_size if community_size > 0 else 0.0
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph, community_metrics: CommunityMetrics) -> GraphMetrics:
        """Calculate graph-level metrics.
        
        Args:
            graph: NetworkX directed graph
            community_metrics: Community detection results
            
        Returns:
            GraphMetrics object with graph-level statistics
        """
        metrics = GraphMetrics()
        
        try:
            # Basic properties
            metrics.num_nodes = graph.number_of_nodes()
            metrics.num_edges = graph.number_of_edges()
            
            if metrics.num_nodes > 1:
                max_edges = metrics.num_nodes * (metrics.num_nodes - 1)
                metrics.density = metrics.num_edges / max_edges if max_edges > 0 else 0.0
            
            # Connectivity
            components = list(nx.weakly_connected_components(graph))
            metrics.num_components = len(components)
            metrics.largest_component_size = max(len(comp) for comp in components) if components else 0
            
            # Clustering
            try:
                metrics.average_clustering = nx.average_clustering(graph.to_undirected())
            except Exception as e:
                logger.warning(f"Failed to calculate average clustering: {e}")
            
            # Path metrics (only for largest component if reasonable size)
            largest_component = max(components, key=len) if components else set()
            if len(largest_component) > 1 and len(largest_component) <= 1000:
                try:
                    subgraph = graph.subgraph(largest_component)
                    undirected_subgraph = subgraph.to_undirected()
                    
                    if nx.is_connected(undirected_subgraph):
                        metrics.diameter = nx.diameter(undirected_subgraph)
                        metrics.average_path_length = nx.average_shortest_path_length(undirected_subgraph)
                except Exception as e:
                    logger.warning(f"Failed to calculate path metrics: {e}")
            
            # Community metrics
            metrics.modularity = community_metrics.modularity
            metrics.num_communities = community_metrics.num_communities
            
            # Centralization metrics
            try:
                degree_values = [d for n, d in graph.degree()]
                if degree_values:
                    max_degree = max(degree_values)
                    sum_diff = sum(max_degree - d for d in degree_values)
                    max_sum_diff = (metrics.num_nodes - 1) * (metrics.num_nodes - 2)
                    metrics.degree_centralization = sum_diff / max_sum_diff if max_sum_diff > 0 else 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate centralization metrics: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
        
        return metrics
    
    def get_top_nodes_by_metric(self, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                               metric_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top nodes by a specific metric.
        
        Args:
            node_metrics: Dictionary of node metrics
            metric_name: Name of the metric to rank by
            top_k: Number of top nodes to return
            
        Returns:
            List of (node_name, metric_value) tuples sorted by metric value
        """
        try:
            node_values = []
            for node, metrics in node_metrics.items():
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    node_values.append((node, value))
            
            # Sort by metric value in descending order
            node_values.sort(key=lambda x: x[1], reverse=True)
            return node_values[:top_k]
        
        except Exception as e:
            logger.error(f"Error getting top nodes by {metric_name}: {e}")
            return []
    
    def get_community_summary(self, community_metrics: CommunityMetrics) -> Dict[str, Any]:
        """Get a summary of community detection results.
        
        Args:
            community_metrics: Community detection results
            
        Returns:
            Dictionary with community summary statistics
        """
        if not community_metrics.communities:
            return {
                'num_communities': 0,
                'modularity': 0.0,
                'average_community_size': 0.0,
                'largest_community_size': 0,
                'smallest_community_size': 0
            }
        
        sizes = list(community_metrics.community_sizes.values())
        
        return {
            'num_communities': community_metrics.num_communities,
            'modularity': community_metrics.modularity,
            'average_community_size': sum(sizes) / len(sizes) if sizes else 0.0,
            'largest_community_size': max(sizes) if sizes else 0,
            'smallest_community_size': min(sizes) if sizes else 0,
            'community_size_distribution': sorted(sizes, reverse=True)
        }
    
    def _identify_bridge_nodes(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                              community_metrics: CommunityMetrics):
        """Identify bridge nodes that connect different communities.
        
        Args:
            graph: NetworkX directed graph
            node_metrics: Dictionary of node metrics to update
            community_metrics: Community detection results
        """
        try:
            if not community_metrics.node_communities:
                logger.warning("No communities detected, skipping bridge node identification")
                return
            
            # Convert to undirected for bridge analysis
            undirected_graph = graph.to_undirected()
            
            # Calculate inter-community connections for each node
            for node in graph.nodes():
                if node not in community_metrics.node_communities:
                    continue
                
                node_community = community_metrics.node_communities[node]
                inter_community_connections = 0
                connected_communities = set()
                
                # Check all neighbors
                for neighbor in undirected_graph.neighbors(node):
                    if neighbor in community_metrics.node_communities:
                        neighbor_community = community_metrics.node_communities[neighbor]
                        if neighbor_community != node_community:
                            inter_community_connections += 1
                            connected_communities.add(neighbor_community)
                
                # Update node metrics
                node_metrics[node].inter_community_connections = inter_community_connections
                
                # Calculate bridge score based on:
                # 1. Number of inter-community connections
                # 2. Number of different communities connected
                # 3. Betweenness centrality (already calculated)
                # 4. Effective size (structural holes measure)
                
                total_connections = undirected_graph.degree(node)
                if total_connections > 0:
                    inter_community_ratio = inter_community_connections / total_connections
                    community_diversity = len(connected_communities)
                    
                    # Combine multiple factors for bridge score
                    bridge_score = (
                        0.3 * inter_community_ratio +  # Proportion of inter-community connections
                        0.2 * min(1.0, community_diversity / 3.0) +  # Diversity of connected communities
                        0.3 * node_metrics[node].betweenness_centrality +  # Betweenness centrality
                        0.2 * min(1.0, node_metrics[node].effective_size / 5.0)  # Effective size (normalized)
                    )
                    
                    node_metrics[node].bridge_score = bridge_score
                    
                    # Mark as bridge node if score is above threshold
                    # A node is considered a bridge if:
                    # 1. It has reasonable bridge score (> 0.25) - lowered threshold further
                    # 2. It connects at least 1 different community (for 2-community graphs)
                    # 3. At least 10% of its connections are inter-community - lowered threshold
                    node_metrics[node].is_bridge_node = (
                        bridge_score > 0.25 and 
                        community_diversity >= 1 and 
                        inter_community_ratio >= 0.1
                    )
            
            # Log bridge node statistics
            bridge_nodes = [node for node, metrics in node_metrics.items() if metrics.is_bridge_node]
            logger.info(f"Identified {len(bridge_nodes)} bridge nodes out of {len(node_metrics)} total nodes")
            
            if bridge_nodes:
                avg_bridge_score = sum(node_metrics[node].bridge_score for node in bridge_nodes) / len(bridge_nodes)
                logger.info(f"Average bridge score: {avg_bridge_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error identifying bridge nodes: {e}")
    
    def _analyze_temporal_patterns(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics]):
        """Analyze temporal patterns for note creation/modification times.
        
        Args:
            graph: NetworkX directed graph
            node_metrics: Dictionary of node metrics to update
        """
        try:
            current_time = datetime.datetime.now().timestamp()
            
            # Extract temporal information from node attributes or file paths
            temporal_data = []
            
            for node in graph.nodes():
                creation_time = None
                modification_time = None
                
                # Try to get timestamps from node attributes
                node_data = graph.nodes[node]
                if isinstance(node_data, dict):
                    creation_time = node_data.get('creation_time')
                    modification_time = node_data.get('modification_time') or node_data.get('modified_time')
                
                # If no timestamps in attributes, try to extract from file path
                if creation_time is None and modification_time is None:
                    try:
                        # Assume node name is a file path
                        file_path = Path(node)
                        if file_path.exists():
                            stat = file_path.stat()
                            creation_time = stat.st_ctime
                            modification_time = stat.st_mtime
                    except (OSError, ValueError):
                        # File doesn't exist or invalid path, skip temporal analysis for this node
                        pass
                
                # Update node metrics with temporal information
                if creation_time is not None:
                    node_metrics[node].creation_time = creation_time
                    temporal_data.append(('creation', creation_time, node))
                
                if modification_time is not None:
                    node_metrics[node].modification_time = modification_time
                    temporal_data.append(('modification', modification_time, node))
            
            if not temporal_data:
                logger.info("No temporal information available for temporal pattern analysis")
                return
            
            # Calculate temporal activity scores
            self._calculate_temporal_activity_scores(node_metrics, temporal_data, current_time)
            
            # Analyze temporal patterns
            self._analyze_temporal_clustering(graph, node_metrics, temporal_data)
            
            logger.info(f"Analyzed temporal patterns for {len([n for n in node_metrics.values() if n.creation_time or n.modification_time])} nodes")
        
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
    
    def _calculate_temporal_activity_scores(self, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                                          temporal_data: List[Tuple[str, float, str]], current_time: float):
        """Calculate temporal activity scores for nodes.
        
        Args:
            node_metrics: Dictionary of node metrics to update
            temporal_data: List of (event_type, timestamp, node) tuples
            current_time: Current timestamp
        """
        try:
            # Sort temporal data by timestamp
            temporal_data.sort(key=lambda x: x[1])
            
            if len(temporal_data) < 2:
                return
            
            # Calculate time spans
            earliest_time = temporal_data[0][1]
            latest_time = temporal_data[-1][1]
            total_time_span = latest_time - earliest_time
            
            if total_time_span <= 0:
                return
            
            # Calculate activity scores for each node
            for node in node_metrics.keys():
                node_events = [(event_type, timestamp) for event_type, timestamp, n in temporal_data if n == node]
                
                if not node_events:
                    continue
                
                # Calculate recency score (more recent = higher score)
                latest_event_time = max(timestamp for _, timestamp in node_events)
                time_since_latest = current_time - latest_event_time
                
                # Normalize recency (1.0 for very recent, 0.0 for very old)
                # Use exponential decay with half-life of 1 year (365 * 24 * 3600 seconds)
                half_life = 365 * 24 * 3600
                recency_score = np.exp(-time_since_latest / half_life)
                
                # Calculate frequency score (more events = higher score)
                frequency_score = min(1.0, len(node_events) / 10.0)  # Normalize to max 10 events
                
                # Calculate temporal spread score (events spread over time = higher score)
                if len(node_events) > 1:
                    event_times = [timestamp for _, timestamp in node_events]
                    event_span = max(event_times) - min(event_times)
                    spread_score = min(1.0, event_span / total_time_span)
                else:
                    spread_score = 0.0
                
                # Combine scores
                activity_score = (
                    0.5 * recency_score +      # Recent activity is most important
                    0.3 * frequency_score +    # Frequency of updates
                    0.2 * spread_score         # Temporal spread
                )
                
                node_metrics[node].temporal_activity_score = activity_score
        
        except Exception as e:
            logger.error(f"Error calculating temporal activity scores: {e}")
    
    def _analyze_temporal_clustering(self, graph: nx.DiGraph, node_metrics: Dict[str, AdvancedCentralityMetrics], 
                                   temporal_data: List[Tuple[str, float, str]]):
        """Analyze temporal clustering patterns in the graph.
        
        Args:
            graph: NetworkX directed graph
            node_metrics: Dictionary of node metrics
            temporal_data: List of (event_type, timestamp, node) tuples
        """
        try:
            # Group nodes by time periods (e.g., months)
            time_periods = defaultdict(list)
            
            for event_type, timestamp, node in temporal_data:
                # Group by month
                dt = datetime.datetime.fromtimestamp(timestamp)
                period_key = f"{dt.year}-{dt.month:02d}"
                time_periods[period_key].append(node)
            
            # Analyze connectivity within and between time periods
            period_connectivity = {}
            
            for period, nodes in time_periods.items():
                if len(nodes) < 2:
                    continue
                
                # Calculate internal connectivity (edges within the period)
                internal_edges = 0
                external_edges = 0
                
                for node in nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor in nodes:
                            internal_edges += 1
                        else:
                            external_edges += 1
                
                # Calculate connectivity metrics
                total_possible_internal = len(nodes) * (len(nodes) - 1)
                internal_density = internal_edges / total_possible_internal if total_possible_internal > 0 else 0.0
                
                period_connectivity[period] = {
                    'nodes': len(nodes),
                    'internal_edges': internal_edges,
                    'external_edges': external_edges,
                    'internal_density': internal_density,
                    'external_ratio': external_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0.0
                }
            
            # Log temporal clustering insights
            if period_connectivity:
                avg_internal_density = sum(p['internal_density'] for p in period_connectivity.values()) / len(period_connectivity)
                logger.info(f"Temporal clustering analysis: {len(period_connectivity)} time periods, "
                           f"average internal density: {avg_internal_density:.3f}")
        
        except Exception as e:
            logger.error(f"Error analyzing temporal clustering: {e}")
    
    def get_bridge_nodes(self, node_metrics: Dict[str, AdvancedCentralityMetrics], top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top bridge nodes by bridge score.
        
        Args:
            node_metrics: Dictionary of node metrics
            top_k: Number of top bridge nodes to return
            
        Returns:
            List of (node_name, bridge_score) tuples sorted by bridge score
        """
        try:
            bridge_nodes = []
            for node, metrics in node_metrics.items():
                if metrics.is_bridge_node:
                    bridge_nodes.append((node, metrics.bridge_score))
            
            # Sort by bridge score in descending order
            bridge_nodes.sort(key=lambda x: x[1], reverse=True)
            return bridge_nodes[:top_k]
        
        except Exception as e:
            logger.error(f"Error getting bridge nodes: {e}")
            return []
    
    def get_temporal_insights(self, node_metrics: Dict[str, AdvancedCentralityMetrics]) -> Dict[str, Any]:
        """Get insights about temporal patterns in the graph.
        
        Args:
            node_metrics: Dictionary of node metrics
            
        Returns:
            Dictionary with temporal insights
        """
        try:
            nodes_with_temporal_data = [
                metrics for metrics in node_metrics.values() 
                if metrics.creation_time is not None or metrics.modification_time is not None
            ]
            
            if not nodes_with_temporal_data:
                return {
                    'nodes_with_temporal_data': 0,
                    'average_activity_score': 0.0,
                    'most_active_nodes': [],
                    'temporal_coverage': 0.0
                }
            
            # Calculate statistics
            activity_scores = [m.temporal_activity_score for m in nodes_with_temporal_data]
            avg_activity = sum(activity_scores) / len(activity_scores) if activity_scores else 0.0
            
            # Get most active nodes
            most_active = []
            for node, metrics in node_metrics.items():
                if metrics.temporal_activity_score > 0:
                    most_active.append((node, metrics.temporal_activity_score))
            most_active.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate temporal coverage
            creation_times = [m.creation_time for m in nodes_with_temporal_data if m.creation_time]
            modification_times = [m.modification_time for m in nodes_with_temporal_data if m.modification_time]
            all_times = creation_times + modification_times
            
            temporal_coverage = 0.0
            if len(all_times) > 1:
                time_span = max(all_times) - min(all_times)
                current_time = datetime.datetime.now().timestamp()
                total_possible_span = current_time - min(all_times)
                temporal_coverage = time_span / total_possible_span if total_possible_span > 0 else 0.0
            
            return {
                'nodes_with_temporal_data': len(nodes_with_temporal_data),
                'average_activity_score': avg_activity,
                'most_active_nodes': most_active[:10],
                'temporal_coverage': temporal_coverage,
                'creation_time_range': (min(creation_times), max(creation_times)) if creation_times else None,
                'modification_time_range': (min(modification_times), max(modification_times)) if modification_times else None
            }
        
        except Exception as e:
            logger.error(f"Error getting temporal insights: {e}")
            return {}