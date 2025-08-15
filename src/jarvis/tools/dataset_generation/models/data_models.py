"""
Core data models for dataset generation.

This module defines the data structures used throughout the dataset generation process,
including note data, features, statistics, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class NoteData:
    """Complete note information for dataset generation."""
    path: str
    title: str
    content: str
    metadata: dict[str, Any]
    tags: list[str]
    outgoing_links: list[str]
    embedding: np.ndarray | None = None
    quality_stage: str | None = None
    word_count: int = 0
    creation_date: datetime | None = None
    last_modified: datetime | None = None
    # Enhanced frontmatter properties
    all_frontmatter_properties: dict[str, Any] = field(default_factory=dict)
    semantic_relationships: dict[str, list[str]] = field(default_factory=dict)
    progress_indicators: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization processing."""
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())

        # Extract dates from metadata if not provided
        if self.creation_date is None and 'created' in self.metadata:
            try:
                timestamp = self.metadata['created']
                if isinstance(timestamp, (int, float)):
                    self.creation_date = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                pass

        if self.last_modified is None and 'modified' in self.metadata:
            try:
                timestamp = self.metadata['modified']
                if isinstance(timestamp, (int, float)):
                    self.last_modified = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                pass


@dataclass
class LinkStatistics:
    """Statistics about extracted links."""
    total_links: int = 0
    unique_links: int = 0
    broken_links: int = 0
    self_links: int = 0
    bidirectional_links: int = 0
    link_types: dict[str, int] = field(default_factory=dict)
    avg_outgoing_links: float = 0.0
    max_outgoing_links: int = 0
    notes_with_no_links: int = 0
    # Error tracking fields
    notes_failed: int = 0
    permission_errors: int = 0
    encoding_errors: int = 0
    corrupted_files: int = 0

    def compute_derived_stats(self, total_notes: int):
        """Compute derived statistics."""
        if total_notes > 0:
            self.avg_outgoing_links = self.total_links / total_notes
            self.notes_with_no_links = total_notes - (self.total_links - self.self_links)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of link extraction."""
        total_processed = self.total_links + self.notes_failed
        if total_processed == 0:
            return 1.0
        return (total_processed - self.notes_failed) / total_processed

    @property
    def link_quality_rate(self) -> float:
        """Calculate quality rate of extracted links."""
        if self.total_links == 0:
            return 1.0
        return (self.total_links - self.broken_links) / self.total_links


@dataclass
class NoteFeatures:
    """Individual note features for dataset."""
    note_path: str
    note_title: str
    word_count: int
    tag_count: int
    quality_stage: str
    creation_date: datetime
    last_modified: datetime
    outgoing_links_count: int
    incoming_links_count: int = 0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank_score: float = 0.0
    clustering_coefficient: float = 0.0
    semantic_cluster_id: int = -1
    semantic_summary: str = ""
    file_size: int = 0
    reading_time_minutes: float = 0.0
    # Enhanced frontmatter-derived features
    aliases_count: int = 0
    domains_count: int = 0
    concepts_count: int = 0
    sources_count: int = 0
    has_summary_field: bool = False
    progress_state: str = "unknown"  # 🌱, 🌿, 🌲, ⚛️, 🗺️, ⧉, 🎓
    semantic_up_links: int = 0
    semantic_similar_links: int = 0
    semantic_leads_to_links: int = 0
    semantic_extends_links: int = 0
    semantic_implements_links: int = 0
    # Content analysis features
    heading_count: int = 0
    max_heading_depth: int = 0
    technical_term_density: float = 0.0
    concept_density_score: float = 0.0
    # TF-IDF features
    top_tfidf_terms: str = ""  # JSON string of top terms and scores
    tfidf_vocabulary_richness: float = 0.0  # Ratio of unique terms to total terms
    avg_tfidf_score: float = 0.0  # Average TF-IDF score for the document
    # Content analysis features
    sentiment_score: float = 0.0  # -1 (negative) to 1 (positive)
    sentiment_label: str = "neutral"  # positive, negative, neutral
    readability_score: float = 0.0  # Flesch Reading Ease (0-100)
    readability_grade: float = 0.0  # Grade level
    named_entities_json: str = ""  # JSON string of named entities
    entity_types_json: str = ""  # JSON string of entity type counts
    content_type: str = "general"  # technical, academic, creative, etc.
    complexity_score: float = 0.0  # 0-1 normalized complexity
    vocabulary_richness: float = 0.0  # unique/total ratio
    # Topic modeling features
    dominant_topic_id: int = -1  # ID of the dominant topic
    dominant_topic_probability: float = 0.0  # Probability of dominant topic
    topic_probabilities_json: str = ""  # JSON string of all topic probabilities
    topic_label: str = ""  # Human-readable topic label
    topic_coherence_score: float = 0.0  # Topic coherence score
    semantic_cluster_id_topic: int = -1  # Cluster ID based on topic features

    def __post_init__(self):
        """Post-initialization processing."""
        # Estimate reading time (average 200 words per minute)
        if self.reading_time_minutes == 0.0 and self.word_count > 0:
            self.reading_time_minutes = self.word_count / 200.0


@dataclass
class PairFeatures:
    """Note pair features for dataset."""
    note_a_path: str
    note_b_path: str
    cosine_similarity: float
    tfidf_similarity: float
    combined_similarity: float
    semantic_cluster_match: bool
    tag_overlap_count: int
    tag_jaccard_similarity: float
    vault_path_distance: int
    shortest_path_length: int
    common_neighbors_count: int
    adamic_adar_score: float
    word_count_ratio: float
    creation_time_diff_days: float
    quality_stage_compatibility: int
    source_centrality: float
    target_centrality: float
    clustering_coefficient: float
    link_exists: bool
    same_folder: bool = False
    content_length_ratio: float = 1.0
    title_similarity: float = 0.0
    # Topic modeling features
    topic_similarity: float = 0.0  # Similarity based on topic probabilities
    same_dominant_topic: bool = False  # Whether both notes have the same dominant topic
    topic_coherence_avg: float = 0.0  # Average topic coherence of both notes

    def __post_init__(self):
        """Post-initialization processing."""
        # Determine if notes are in the same folder
        from pathlib import Path
        path_a = Path(self.note_a_path)
        path_b = Path(self.note_b_path)
        self.same_folder = path_a.parent == path_b.parent

        # Normalize ratios to be between 0 and 1
        if self.word_count_ratio > 1.0:
            self.word_count_ratio = 1.0 / self.word_count_ratio

        if self.content_length_ratio > 1.0:
            self.content_length_ratio = 1.0 / self.content_length_ratio


@dataclass
class ValidationResult:
    """Result of data validation operations."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes_processed: int = 0
    notes_failed: int = 0
    links_extracted: int = 0
    links_broken: int = 0
    
    # Enhanced fields for Areas/ filtering
    areas_folder_exists: bool = False
    areas_notes_count: int = 0
    filtering_mode: str = "full_vault"  # "areas_only" or "full_vault"
    excluded_notes_count: int = 0
    areas_folder_path: str | None = None
    areas_folder_name: str | None = None
    areas_subdirectory_count: int = 0
    areas_total_size_bytes: int = 0
    min_content_threshold: int = 5
    areas_validation_passed: bool = False

    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing."""
        if self.notes_processed == 0:
            return 0.0
        return (self.notes_processed - self.notes_failed) / self.notes_processed

    @property
    def link_quality_rate(self) -> float:
        """Calculate quality rate of extracted links."""
        if self.links_extracted == 0:
            return 1.0
        return (self.links_extracted - self.links_broken) / self.links_extracted


@dataclass
class GenerationSummary:
    """Summary of dataset generation process."""
    total_notes: int
    notes_processed: int
    notes_failed: int
    pairs_generated: int
    positive_pairs: int
    negative_pairs: int
    total_time_seconds: float
    link_statistics: LinkStatistics
    validation_result: ValidationResult
    output_files: dict[str, str] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    
    # Enhanced fields for filtering metadata
    filtering_enabled: bool = False
    areas_folder_path: str | None = None
    excluded_folders: list[str] = field(default_factory=list)
    privacy_mode: bool = False
    
    # Additional filtering metadata for comprehensive reporting
    filtering_summary: str | None = None
    content_protection_level: str = "none"  # "none", "medium", "high", "unknown"
    privacy_message: str | None = None
    excluded_folder_count: int = 0
    areas_notes_count: int = 0
    total_vault_notes: int | None = None  # For comparison when filtering is enabled
    privacy_protection_percentage: float | None = None  # Percentage of content excluded

    @property
    def processing_rate(self) -> float:
        """Calculate notes processed per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.notes_processed / self.total_time_seconds

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_notes == 0:
            return 0.0
        return self.notes_processed / self.total_notes

    @property
    def positive_ratio(self) -> float:
        """Calculate ratio of positive to total pairs."""
        if self.pairs_generated == 0:
            return 0.0
        return self.positive_pairs / self.pairs_generated


@dataclass
class DatasetGenerationResult:
    """Complete result of dataset generation process."""
    success: bool
    summary: GenerationSummary
    notes_dataset_path: str | None = None
    pairs_dataset_path: str | None = None
    error_message: str | None = None
    intermediate_data: dict[str, Any] = field(default_factory=dict)

    @property
    def datasets_created(self) -> list[str]:
        """Get list of successfully created dataset files."""
        datasets = []
        if self.notes_dataset_path:
            datasets.append(self.notes_dataset_path)
        if self.pairs_dataset_path:
            datasets.append(self.pairs_dataset_path)
        return datasets


@dataclass
class CentralityMetrics:
    """Graph centrality metrics for a note."""
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank_score: float = 0.0
    degree_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            'betweenness_centrality': self.betweenness_centrality,
            'closeness_centrality': self.closeness_centrality,
            'pagerank_score': self.pagerank_score,
            'degree_centrality': self.degree_centrality,
            'eigenvector_centrality': self.eigenvector_centrality,
            'clustering_coefficient': self.clustering_coefficient
        }


@dataclass
class Link:
    """Represents a link between notes."""
    source: str
    target: str
    link_type: str = "wikilink"
    display_text: str | None = None
    is_valid: bool = True
    line_number: int | None = None

    def __hash__(self) -> int:
        """Make Link hashable for use in sets."""
        return hash((self.source, self.target, self.link_type))

    def __eq__(self, other) -> bool:
        """Compare links for equality."""
        if not isinstance(other, Link):
            return False
        return (self.source == other.source and
                self.target == other.target and
                self.link_type == other.link_type)


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    start_time: datetime
    end_time: datetime | None = None
    items_processed: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    bytes_processed: int = 0
    memory_usage_mb: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Calculate processing duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        duration = self.duration_seconds
        if duration == 0:
            return 0.0
        return self.items_processed / duration

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.items_processed + self.items_failed + self.items_skipped
        if total == 0:
            return 1.0
        return self.items_processed / total
