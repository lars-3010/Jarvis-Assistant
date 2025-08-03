"""
Interface definitions for dataset generation components.

This module defines the protocols and abstract base classes that establish
contracts for dataset generation components, enabling testability and
loose coupling.
"""

from abc import ABC, abstractmethod
from typing import Protocol

import networkx as nx
import numpy as np
import pandas as pd

from .data_models import (
    CentralityMetrics,
    Link,
    NoteData,
    NoteFeatures,
    PairFeatures,
    ValidationResult,
)


class IDatasetGenerator(Protocol):
    """Interface for dataset generation components."""

    def generate_dataset(self, **kwargs) -> pd.DataFrame:
        """Generate dataset with specified parameters.
        
        Args:
            **kwargs: Implementation-specific parameters
            
        Returns:
            Generated dataset as pandas DataFrame
        """
        ...

    def validate_inputs(self, **kwargs) -> ValidationResult:
        """Validate input parameters and data quality.
        
        Args:
            **kwargs: Implementation-specific parameters
            
        Returns:
            Validation result with errors and warnings
        """
        ...

    def get_feature_descriptions(self) -> dict[str, str]:
        """Get descriptions of all generated features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        ...


class ILinkExtractor(Protocol):
    """Interface for link extraction components."""

    def extract_links_from_content(self, content: str, source_path: str) -> list[Link]:
        """Extract links from note content.
        
        Args:
            content: Note content to extract links from
            source_path: Path of the source note
            
        Returns:
            List of extracted links
        """
        ...

    def build_link_graph(self, notes: dict[str, str]) -> nx.DiGraph:
        """Build complete link graph from all notes.
        
        Args:
            notes: Dictionary mapping note paths to content
            
        Returns:
            Directed graph representing note relationships
        """
        ...

    def validate_links(self, links: list[Link]) -> ValidationResult:
        """Validate extracted links against vault structure.
        
        Args:
            links: List of links to validate
            
        Returns:
            Validation result with broken link information
        """
        ...


class IFeatureExtractor(Protocol):
    """Interface for feature extraction components."""

    def extract_note_features(self, note_data: NoteData, graph: nx.DiGraph) -> NoteFeatures:
        """Extract all features for a single note.
        
        Args:
            note_data: Complete note information
            graph: Link graph for centrality calculations
            
        Returns:
            Extracted note features
        """
        ...

    def extract_pair_features(self, note_a: NoteData, note_b: NoteData,
                            graph: nx.DiGraph) -> PairFeatures:
        """Extract all features for a note pair.
        
        Args:
            note_a: First note data
            note_b: Second note data
            graph: Link graph for relationship calculations
            
        Returns:
            Extracted pair features
        """
        ...


class ISamplingStrategy(Protocol):
    """Interface for negative sampling strategies."""

    def sample_negative_pairs(self, positive_pairs: set[tuple[str, str]],
                            all_notes: list[str], target_count: int) -> list[tuple[str, str]]:
        """Sample negative examples according to strategy.
        
        Args:
            positive_pairs: Set of existing positive pairs
            all_notes: List of all available notes
            target_count: Number of negative samples to generate
            
        Returns:
            List of sampled negative pairs
        """
        ...

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy.
        
        Returns:
            Strategy name
        """
        ...


class IProgressTracker(Protocol):
    """Interface for progress tracking during dataset generation."""

    def start_operation(self, operation_name: str, total_items: int) -> str:
        """Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
            total_items: Total number of items to process
            
        Returns:
            Operation ID for tracking
        """
        ...

    def update_progress(self, operation_id: str, processed_items: int) -> None:
        """Update progress for an operation.
        
        Args:
            operation_id: ID of the operation
            processed_items: Number of items processed so far
        """
        ...

    def finish_operation(self, operation_id: str) -> None:
        """Mark an operation as finished.
        
        Args:
            operation_id: ID of the operation
        """
        ...

    def get_estimated_time_remaining(self, operation_id: str) -> float | None:
        """Get estimated time remaining for an operation.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            Estimated seconds remaining, or None if unknown
        """
        ...


class IValidator(Protocol):
    """Interface for data validation components."""

    def validate_vault_structure(self, vault_path: str) -> ValidationResult:
        """Validate vault structure and accessibility.
        
        Args:
            vault_path: Path to the vault to validate
            
        Returns:
            Validation result
        """
        ...

    def validate_dataset_quality(self, dataset: pd.DataFrame,
                               dataset_type: str) -> ValidationResult:
        """Validate generated dataset quality.
        
        Args:
            dataset: Generated dataset to validate
            dataset_type: Type of dataset ('notes' or 'pairs')
            
        Returns:
            Validation result
        """
        ...


# Abstract base classes for concrete implementations

class BaseSamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, random_seed: int | None = None):
        """Initialize sampling strategy.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed

    @abstractmethod
    def sample_negative_pairs(self, positive_pairs: set[tuple[str, str]],
                            all_notes: list[str], target_count: int) -> list[tuple[str, str]]:
        """Sample negative examples according to strategy."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy."""
        pass


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    def __init__(self):
        """Initialize feature extractor."""
        pass

    @abstractmethod
    def extract_note_features(self, note_data: NoteData, graph: nx.DiGraph) -> NoteFeatures:
        """Extract all features for a single note."""
        pass

    @abstractmethod
    def extract_pair_features(self, note_a: NoteData, note_b: NoteData,
                            graph: nx.DiGraph) -> PairFeatures:
        """Extract all features for a note pair."""
        pass

    def compute_centrality_metrics(self, node: str, graph: nx.DiGraph) -> CentralityMetrics:
        """Compute centrality metrics for a node in the graph.
        
        Args:
            node: Node identifier
            graph: Network graph
            
        Returns:
            Centrality metrics for the node
        """
        metrics = CentralityMetrics()

        if not graph.has_node(node):
            return metrics

        try:
            # Compute basic centrality measures
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            pagerank = nx.pagerank(graph)
            degree = nx.degree_centrality(graph)

            metrics.betweenness_centrality = betweenness.get(node, 0.0)
            metrics.closeness_centrality = closeness.get(node, 0.0)
            metrics.pagerank_score = pagerank.get(node, 0.0)
            metrics.degree_centrality = degree.get(node, 0.0)

            # Compute clustering coefficient
            if graph.is_directed():
                # Convert to undirected for clustering coefficient
                undirected = graph.to_undirected()
                clustering = nx.clustering(undirected)
                metrics.clustering_coefficient = clustering.get(node, 0.0)
            else:
                clustering = nx.clustering(graph)
                metrics.clustering_coefficient = clustering.get(node, 0.0)

            # Compute eigenvector centrality (may fail for some graphs)
            try:
                eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
                metrics.eigenvector_centrality = eigenvector.get(node, 0.0)
            except (nx.NetworkXError, np.linalg.LinAlgError):
                metrics.eigenvector_centrality = 0.0

        except Exception:
            # Return default metrics if computation fails
            pass

        return metrics
