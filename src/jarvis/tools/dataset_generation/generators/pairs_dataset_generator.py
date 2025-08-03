"""
Pairs dataset generator for note comparison analysis.

This module creates a comprehensive dataset containing comparative features
between pairs of notes, including semantic similarity, graph-based features,
and smart negative sampling for link prediction modeling.
"""

import math
import random
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from jarvis.services.graph.database import GraphDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.logging import setup_logging

from ..models.data_models import NoteData, PairFeatures, ProcessingStats, ValidationResult
from ..models.exceptions import FeatureEngineeringError, InsufficientDataError, SamplingError
from ..models.interfaces import BaseSamplingStrategy

logger = setup_logging(__name__)


class RandomSamplingStrategy(BaseSamplingStrategy):
    """Random negative sampling strategy."""

    def sample_negative_pairs(self, positive_pairs: set[tuple[str, str]],
                            all_notes: list[str], target_count: int) -> list[tuple[str, str]]:
        """Sample negative examples randomly."""
        if self.random_seed is not None:
            random.seed(self.random_seed)

        negative_pairs = []
        max_attempts = target_count * 10  # Prevent infinite loops
        attempts = 0

        while len(negative_pairs) < target_count and attempts < max_attempts:
            # Sample two different notes randomly
            note_a = random.choice(all_notes)
            note_b = random.choice(all_notes)

            if note_a != note_b:
                pair = tuple(sorted([note_a, note_b]))
                if pair not in positive_pairs and pair not in negative_pairs:
                    negative_pairs.append(pair)

            attempts += 1

        return negative_pairs

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy."""
        return "random"


class StratifiedSamplingStrategy(BaseSamplingStrategy):
    """Stratified negative sampling based on note properties."""

    def __init__(self, note_data: dict[str, NoteData], random_seed: int | None = None):
        """Initialize with note data for stratification."""
        super().__init__(random_seed)
        self.note_data = note_data
        self._folder_groups = self._group_by_folder()
        self._tag_groups = self._group_by_tags()

    def _group_by_folder(self) -> dict[str, list[str]]:
        """Group notes by folder."""
        groups = {}
        for note_path in self.note_data:
            folder = str(Path(note_path).parent)
            if folder not in groups:
                groups[folder] = []
            groups[folder].append(note_path)
        return groups

    def _group_by_tags(self) -> dict[str, list[str]]:
        """Group notes by common tags."""
        groups = {}
        for note_path, note_data in self.note_data.items():
            for tag in note_data.tags:
                if tag not in groups:
                    groups[tag] = []
                groups[tag].append(note_path)
        return groups

    def sample_negative_pairs(self, positive_pairs: set[tuple[str, str]],
                            all_notes: list[str], target_count: int) -> list[tuple[str, str]]:
        """Sample negative examples using stratified approach."""
        if self.random_seed is not None:
            random.seed(self.random_seed)

        negative_pairs = []
        max_attempts = target_count * 10
        attempts = 0

        # Stratified sampling: mix of same-folder, same-tag, and random pairs
        same_folder_count = target_count // 3
        same_tag_count = target_count // 3
        random_count = target_count - same_folder_count - same_tag_count

        # Sample same-folder pairs
        negative_pairs.extend(
            self._sample_from_groups(
                self._folder_groups, positive_pairs, same_folder_count
            )
        )

        # Sample same-tag pairs
        negative_pairs.extend(
            self._sample_from_groups(
                self._tag_groups, positive_pairs, same_tag_count
            )
        )

        # Fill remaining with random sampling
        while len(negative_pairs) < target_count and attempts < max_attempts:
            note_a = random.choice(all_notes)
            note_b = random.choice(all_notes)

            if note_a != note_b:
                pair = tuple(sorted([note_a, note_b]))
                if pair not in positive_pairs and pair not in negative_pairs:
                    negative_pairs.append(pair)

            attempts += 1

        return negative_pairs[:target_count]

    def _sample_from_groups(self, groups: dict[str, list[str]],
                          positive_pairs: set[tuple[str, str]],
                          target_count: int) -> list[tuple[str, str]]:
        """Sample negative pairs from within groups."""
        pairs = []
        max_attempts = target_count * 5
        attempts = 0

        group_names = list(groups.keys())

        while len(pairs) < target_count and attempts < max_attempts:
            # Select a group with at least 2 notes
            group_name = random.choice(group_names)
            group_notes = groups[group_name]

            if len(group_notes) >= 2:
                note_a = random.choice(group_notes)
                note_b = random.choice(group_notes)

                if note_a != note_b:
                    pair = tuple(sorted([note_a, note_b]))
                    if pair not in positive_pairs and pair not in pairs:
                        pairs.append(pair)

            attempts += 1

        return pairs

    def get_strategy_name(self) -> str:
        """Get the name of the sampling strategy."""
        return "stratified"


class PairsDatasetGenerator:
    """Generate note pairs comparison dataset."""

    def __init__(self, vector_encoder: VectorEncoder,
                 graph_database: GraphDatabase | None = None,
                 sampling_strategy: BaseSamplingStrategy | None = None):
        """Initialize the pairs dataset generator.
        
        Args:
            vector_encoder: VectorEncoder service for similarity computation
            graph_database: Optional GraphDatabase service for graph features
            sampling_strategy: Strategy for negative sampling (default: random)
        """
        self.vector_encoder = vector_encoder
        self.graph_database = graph_database
        self.sampling_strategy = sampling_strategy or RandomSamplingStrategy()
        self._processing_stats = None

    def generate_dataset(self, notes_data: dict[str, NoteData],
                        link_graph: nx.DiGraph,
                        negative_sampling_ratio: float = 5.0,
                        max_pairs_per_note: int = 1000,
                        batch_size: int = 32,
                        progress_callback=None) -> pd.DataFrame:
        """Generate comprehensive pairs dataset with smart sampling.
        
        Args:
            notes_data: Dictionary mapping note paths to NoteData
            link_graph: NetworkX graph of note relationships
            negative_sampling_ratio: Ratio of negative to positive examples
            max_pairs_per_note: Maximum pairs to consider per note
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame containing pair features
            
        Raises:
            InsufficientDataError: If not enough data for meaningful pairs
            SamplingError: If sampling fails
        """
        logger.info(f"Starting pairs dataset generation for {len(notes_data)} notes")

        if len(notes_data) < 5:
            raise InsufficientDataError(
                f"Insufficient notes for pairs generation: {len(notes_data)} < 5",
                required_minimum=5,
                actual_count=len(notes_data)
            )

        # Initialize processing stats
        self._processing_stats = ProcessingStats(start_time=datetime.now())

        try:
            # Extract positive pairs from the graph
            positive_pairs = self._extract_positive_pairs(link_graph)
            logger.info(f"Found {len(positive_pairs)} positive pairs")

            if len(positive_pairs) == 0:
                raise InsufficientDataError(
                    "No positive pairs found in link graph",
                    required_minimum=1,
                    actual_count=0
                )

            # Generate negative pairs
            all_notes = list(notes_data.keys())
            target_negative_count = min(
                int(len(positive_pairs) * negative_sampling_ratio),
                (len(all_notes) * (len(all_notes) - 1)) // 2 - len(positive_pairs)
            )

            logger.info(f"Generating {target_negative_count} negative pairs")
            negative_pairs = self._smart_negative_sampling(
                positive_pairs, all_notes, target_negative_count, notes_data
            )

            # Combine positive and negative pairs
            all_pairs = [(pair[0], pair[1], True) for pair in positive_pairs]
            all_pairs.extend([(pair[0], pair[1], False) for pair in negative_pairs])

            logger.info(f"Total pairs to process: {len(all_pairs)} ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")

            # Limit pairs per note if specified
            if max_pairs_per_note > 0:
                all_pairs = self._limit_pairs_per_note(all_pairs, max_pairs_per_note)
                logger.info(f"Limited to {len(all_pairs)} pairs after per-note limit")

            # Process pairs in batches
            pair_features_list = []
            processed_count = 0
            failed_count = 0

            for i in range(0, len(all_pairs), batch_size):
                batch = all_pairs[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} pairs")

                batch_features = self._process_pair_batch(batch, notes_data, link_graph)

                # Update counters
                for features in batch_features:
                    if features is not None:
                        pair_features_list.append(features)
                        processed_count += 1
                    else:
                        failed_count += 1

                # Update progress
                total_processed = processed_count + failed_count
                if progress_callback:
                    progress_callback(total_processed, len(all_pairs))

                # Log progress periodically
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {total_processed}/{len(all_pairs)} pairs")

            # Update processing stats
            self._processing_stats.end_time = datetime.now()
            self._processing_stats.items_processed = processed_count
            self._processing_stats.items_failed = failed_count

            logger.info(f"Pairs processing complete: {processed_count} successful, {failed_count} failed")

            if not pair_features_list:
                raise FeatureEngineeringError("No pair features could be extracted")

            # Convert to DataFrame
            dataset = self._create_dataframe(pair_features_list)

            logger.info(f"Pairs dataset created: {len(dataset)} rows, {len(dataset.columns)} columns")
            return dataset

        except Exception as e:
            logger.error(f"Pairs dataset generation failed: {e}")
            if isinstance(e, (InsufficientDataError, SamplingError, FeatureEngineeringError)):
                raise
            raise FeatureEngineeringError(f"Failed to generate pairs dataset: {e}") from e

    def _extract_positive_pairs(self, link_graph: nx.DiGraph) -> set[tuple[str, str]]:
        """Extract positive pairs from the link graph.
        
        Args:
            link_graph: NetworkX directed graph
            
        Returns:
            Set of positive pairs (bidirectional)
        """
        positive_pairs = set()

        for source, target in link_graph.edges():
            # Create bidirectional pairs for undirected similarity
            pair = tuple(sorted([source, target]))
            positive_pairs.add(pair)

        return positive_pairs

    def _smart_negative_sampling(self, positive_pairs: set[tuple[str, str]],
                               all_notes: list[str], target_count: int,
                               notes_data: dict[str, NoteData]) -> list[tuple[str, str]]:
        """Implement intelligent negative example sampling.
        
        Args:
            positive_pairs: Set of existing positive pairs
            all_notes: List of all available notes
            target_count: Number of negative samples to generate
            notes_data: Note data for stratified sampling
            
        Returns:
            List of negative pairs
            
        Raises:
            SamplingError: If sampling fails
        """
        try:
            # Use stratified sampling if we have note data
            if isinstance(self.sampling_strategy, StratifiedSamplingStrategy):
                # Already initialized with note_data
                pass
            elif hasattr(self.sampling_strategy, 'note_data'):
                # Update note data for stratified sampling
                self.sampling_strategy.note_data = notes_data
            else:
                # Use existing sampling strategy (e.g., random)
                pass

            negative_pairs = self.sampling_strategy.sample_negative_pairs(
                positive_pairs, all_notes, target_count
            )

            if len(negative_pairs) == 0:
                raise SamplingError(
                    "Failed to generate any negative samples",
                    sampling_strategy=self.sampling_strategy.get_strategy_name(),
                    target_ratio=target_count / len(positive_pairs) if positive_pairs else 0
                )

            logger.info(f"Generated {len(negative_pairs)} negative pairs using {self.sampling_strategy.get_strategy_name()} strategy")
            return negative_pairs

        except Exception as e:
            logger.error(f"Negative sampling failed: {e}")
            raise SamplingError(f"Negative sampling failed: {e}") from e

    def _limit_pairs_per_note(self, all_pairs: list[tuple[str, str, bool]],
                            max_pairs: int) -> list[tuple[str, str, bool]]:
        """Limit the number of pairs per note.
        
        Args:
            all_pairs: List of (note_a, note_b, is_positive) tuples
            max_pairs: Maximum pairs per note
            
        Returns:
            Filtered list of pairs
        """
        note_pair_counts = {}
        filtered_pairs = []

        # Randomize order to ensure fair sampling
        pairs_copy = all_pairs.copy()
        random.shuffle(pairs_copy)

        for note_a, note_b, is_positive in pairs_copy:
            # Count pairs for both notes
            count_a = note_pair_counts.get(note_a, 0)
            count_b = note_pair_counts.get(note_b, 0)

            # Only include if both notes are under the limit
            if count_a < max_pairs and count_b < max_pairs:
                filtered_pairs.append((note_a, note_b, is_positive))
                note_pair_counts[note_a] = count_a + 1
                note_pair_counts[note_b] = count_b + 1

        return filtered_pairs

    def _process_pair_batch(self, batch_pairs: list[tuple[str, str, bool]],
                          notes_data: dict[str, NoteData],
                          link_graph: nx.DiGraph) -> list[PairFeatures | None]:
        """Process a batch of pairs to extract features.
        
        Args:
            batch_pairs: List of (note_a, note_b, is_positive) tuples
            notes_data: Dictionary of note data
            link_graph: NetworkX graph for relationship calculations
            
        Returns:
            List of PairFeatures objects (None for failed pairs)
        """
        batch_features = []

        for note_a_path, note_b_path, link_exists in batch_pairs:
            try:
                # Get note data
                note_a = notes_data.get(note_a_path)
                note_b = notes_data.get(note_b_path)

                if note_a is None or note_b is None:
                    logger.warning(f"Missing note data for pair: {note_a_path}, {note_b_path}")
                    batch_features.append(None)
                    continue

                # Extract pair features
                features = self._compute_pair_features(note_a, note_b, link_graph, link_exists)
                batch_features.append(features)

            except Exception as e:
                logger.warning(f"Failed to extract features for pair {note_a_path}, {note_b_path}: {e}")
                batch_features.append(None)

        return batch_features

    def _compute_pair_features(self, note_a: NoteData, note_b: NoteData,
                             graph: nx.DiGraph, link_exists: bool) -> PairFeatures:
        """Compute all features for a note pair with comprehensive error handling.
        
        Args:
            note_a: First note data
            note_b: Second note data
            graph: NetworkX graph for relationship calculations
            link_exists: Whether a link exists between the notes
            
        Returns:
            PairFeatures object with computed features (with defaults for failed computations)
        """
        feature_errors = []
        
        # Compute semantic similarity with error handling
        cosine_sim = 0.0
        try:
            cosine_sim = self._compute_semantic_similarity(note_a, note_b)
        except Exception as e:
            logger.warning(f"Failed to compute semantic similarity for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("semantic_similarity")

        # Compute tag-based features with error handling
        tag_overlap = 0
        tag_jaccard = 0.0
        try:
            note_a_tags = set(note_a.tags) if note_a.tags else set()
            note_b_tags = set(note_b.tags) if note_b.tags else set()
            tag_overlap = len(note_a_tags & note_b_tags)
            all_tags = note_a_tags | note_b_tags
            tag_jaccard = tag_overlap / len(all_tags) if all_tags else 0.0
        except Exception as e:
            logger.warning(f"Failed to compute tag features for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("tag_features")

        # Compute path-based features with error handling
        path_distance = 0
        try:
            path_distance = self._compute_vault_path_distance(note_a.path, note_b.path)
        except Exception as e:
            logger.warning(f"Failed to compute path distance for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("path_distance")

        # Compute graph-based features with error handling
        shortest_path_length = -1  # -1 indicates no path
        try:
            shortest_path_length = self._compute_shortest_path_length(
                note_a.path, note_b.path, graph
            )
        except Exception as e:
            logger.debug(f"Failed to compute shortest path for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("shortest_path")

        common_neighbors = 0
        try:
            common_neighbors = self._compute_common_neighbors_count(
                note_a.path, note_b.path, graph
            )
        except Exception as e:
            logger.debug(f"Failed to compute common neighbors for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("common_neighbors")

        adamic_adar = 0.0
        try:
            adamic_adar = self._compute_adamic_adar_score(
                note_a.path, note_b.path, graph
            )
        except Exception as e:
            logger.debug(f"Failed to compute Adamic-Adar score for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("adamic_adar")

        # Compute content-based features with error handling
        word_count_ratio = 0.0
        try:
            word_count_a = note_a.word_count if note_a.word_count else 0
            word_count_b = note_b.word_count if note_b.word_count else 0
            max_count = max(word_count_a, word_count_b)
            min_count = min(word_count_a, word_count_b)
            word_count_ratio = min_count / max_count if max_count > 0 else 0.0
        except Exception as e:
            logger.debug(f"Failed to compute word count ratio for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("word_count_ratio")

        # Compute temporal features with error handling
        creation_time_diff = 0.0
        try:
            if note_a.creation_date and note_b.creation_date:
                creation_time_diff = abs((note_a.creation_date - note_b.creation_date).days)
        except Exception as e:
            logger.debug(f"Failed to compute creation time diff for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("creation_time_diff")

        # Compute quality stage compatibility with error handling
        quality_compatibility = 0
        try:
            if (note_a.quality_stage and note_b.quality_stage and 
                note_a.quality_stage != 'unknown' and note_b.quality_stage != 'unknown'):
                quality_compatibility = 1 if note_a.quality_stage == note_b.quality_stage else 0
        except Exception as e:
            logger.debug(f"Failed to compute quality compatibility for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("quality_compatibility")

        # Compute centrality features with error handling
        source_centrality = 0.0
        target_centrality = 0.0
        try:
            if graph and graph.has_node(note_a.path):
                source_centrality = graph.in_degree(note_a.path) + graph.out_degree(note_a.path)
            if graph and graph.has_node(note_b.path):
                target_centrality = graph.in_degree(note_b.path) + graph.out_degree(note_b.path)
        except Exception as e:
            logger.debug(f"Failed to compute centrality features for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("centrality_features")

        # Compute clustering coefficient with error handling
        clustering_coeff = 0.0
        try:
            if graph and graph.has_node(note_a.path) and graph.has_node(note_b.path):
                undirected = graph.to_undirected()
                clustering = nx.clustering(undirected)
                clustering_coeff = (clustering.get(note_a.path, 0.0) + clustering.get(note_b.path, 0.0)) / 2.0
        except Exception as e:
            logger.debug(f"Failed to compute clustering coefficient for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("clustering_coefficient")

        # Compute title similarity with error handling
        title_sim = 0.0
        try:
            title_sim = self._compute_title_similarity(note_a.title, note_b.title)
        except Exception as e:
            logger.debug(f"Failed to compute title similarity for {note_a.path} <-> {note_b.path}: {e}")
            feature_errors.append("title_similarity")

        # Create PairFeatures object with error handling
        try:
            features = PairFeatures(
                note_a_path=note_a.path,
                note_b_path=note_b.path,
                cosine_similarity=cosine_sim,
                semantic_cluster_match=False,  # Would require clustering analysis
                tag_overlap_count=tag_overlap,
                tag_jaccard_similarity=tag_jaccard,
                vault_path_distance=path_distance,
                shortest_path_length=shortest_path_length,
                common_neighbors_count=common_neighbors,
                adamic_adar_score=adamic_adar,
                word_count_ratio=word_count_ratio,
                creation_time_diff_days=creation_time_diff,
                quality_stage_compatibility=quality_compatibility,
                source_centrality=source_centrality,
                target_centrality=target_centrality,
                clustering_coefficient=clustering_coeff,
                link_exists=link_exists,
                title_similarity=title_sim
            )

            # Log feature computation summary
            if feature_errors:
                logger.debug(f"Pair feature computation for {note_a.path} <-> {note_b.path} "
                           f"completed with {len(feature_errors)} errors: {feature_errors}")

            return features

        except Exception as e:
            logger.error(f"Failed to create PairFeatures object for {note_a.path} <-> {note_b.path}: {e}")
            raise FeatureEngineeringError(f"Failed to create pair features: {e}") from e

    def _compute_semantic_similarity(self, note_a: NoteData, note_b: NoteData) -> float:
        """Compute cosine similarity between note embeddings.
        
        Args:
            note_a: First note data
            note_b: Second note data
            
        Returns:
            Cosine similarity score
        """
        if note_a.embedding is None or note_b.embedding is None:
            return 0.0

        try:
            # Reshape for sklearn cosine_similarity
            embedding_a = note_a.embedding.reshape(1, -1)
            embedding_b = note_b.embedding.reshape(1, -1)

            similarity = cosine_similarity(embedding_a, embedding_b)[0, 0]
            return float(similarity)

        except Exception as e:
            logger.warning(f"Failed to compute semantic similarity: {e}")
            return 0.0

    def _compute_vault_path_distance(self, path_a: str, path_b: str) -> int:
        """Compute directory distance between two paths.
        
        Args:
            path_a: First path
            path_b: Second path
            
        Returns:
            Directory distance
        """
        try:
            path_a_parts = Path(path_a).parts
            path_b_parts = Path(path_b).parts

            # Find common prefix
            common_parts = 0
            for a_part, b_part in zip(path_a_parts, path_b_parts, strict=False):
                if a_part == b_part:
                    common_parts += 1
                else:
                    break

            # Calculate distance
            distance = (len(path_a_parts) - common_parts) + (len(path_b_parts) - common_parts)
            return distance

        except Exception:
            return 999  # Large distance for error cases

    def _compute_shortest_path_length(self, note_a: str, note_b: str,
                                    graph: nx.DiGraph) -> int:
        """Compute shortest path length between two notes in the graph.
        
        Args:
            note_a: First note path
            note_b: Second note path
            graph: NetworkX graph
            
        Returns:
            Shortest path length (999 if no path exists)
        """
        if not graph.has_node(note_a) or not graph.has_node(note_b):
            return 999

        try:
            # Convert to undirected for path calculation
            undirected = graph.to_undirected()
            path_length = nx.shortest_path_length(undirected, note_a, note_b)
            return path_length

        except nx.NetworkXNoPath:
            return 999
        except Exception:
            return 999

    def _compute_common_neighbors_count(self, note_a: str, note_b: str,
                                      graph: nx.DiGraph) -> int:
        """Compute number of common neighbors.
        
        Args:
            note_a: First note path
            note_b: Second note path
            graph: NetworkX graph
            
        Returns:
            Number of common neighbors
        """
        if not graph.has_node(note_a) or not graph.has_node(note_b):
            return 0

        try:
            # Convert to undirected for neighbor calculation
            undirected = graph.to_undirected()
            neighbors_a = set(undirected.neighbors(note_a))
            neighbors_b = set(undirected.neighbors(note_b))

            return len(neighbors_a & neighbors_b)

        except Exception:
            return 0

    def _compute_adamic_adar_score(self, note_a: str, note_b: str,
                                 graph: nx.DiGraph) -> float:
        """Compute Adamic-Adar score for the pair.
        
        Args:
            note_a: First note path
            note_b: Second note path
            graph: NetworkX graph
            
        Returns:
            Adamic-Adar score
        """
        if not graph.has_node(note_a) or not graph.has_node(note_b):
            return 0.0

        try:
            # Convert to undirected for calculation
            undirected = graph.to_undirected()
            neighbors_a = set(undirected.neighbors(note_a))
            neighbors_b = set(undirected.neighbors(note_b))
            common_neighbors = neighbors_a & neighbors_b

            if not common_neighbors:
                return 0.0

            score = 0.0
            for neighbor in common_neighbors:
                degree = undirected.degree(neighbor)
                if degree > 1:
                    score += 1.0 / math.log(degree)

            return score

        except Exception:
            return 0.0

    def _compute_title_similarity(self, title_a: str, title_b: str) -> float:
        """Compute title similarity using Jaccard coefficient on words.
        
        Args:
            title_a: First title
            title_b: Second title
            
        Returns:
            Jaccard similarity of title words
        """
        if not title_a or not title_b:
            return 0.0

        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union) if union else 0.0

    def _create_dataframe(self, features_list: list[PairFeatures]) -> pd.DataFrame:
        """Create pandas DataFrame from list of PairFeatures.
        
        Args:
            features_list: List of PairFeatures objects
            
        Returns:
            DataFrame with pair features
        """
        data = []

        for features in features_list:
            row = {
                'note_a_path': features.note_a_path,
                'note_b_path': features.note_b_path,
                'cosine_similarity': features.cosine_similarity,
                'semantic_cluster_match': features.semantic_cluster_match,
                'tag_overlap_count': features.tag_overlap_count,
                'tag_jaccard_similarity': features.tag_jaccard_similarity,
                'vault_path_distance': features.vault_path_distance,
                'shortest_path_length': features.shortest_path_length,
                'common_neighbors_count': features.common_neighbors_count,
                'adamic_adar_score': features.adamic_adar_score,
                'word_count_ratio': features.word_count_ratio,
                'creation_time_diff_days': features.creation_time_diff_days,
                'quality_stage_compatibility': features.quality_stage_compatibility,
                'source_centrality': features.source_centrality,
                'target_centrality': features.target_centrality,
                'clustering_coefficient': features.clustering_coefficient,
                'link_exists': features.link_exists,
                'same_folder': features.same_folder,
                'content_length_ratio': features.content_length_ratio,
                'title_similarity': features.title_similarity
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Ensure proper data types
        numeric_columns = [
            'cosine_similarity', 'tag_overlap_count', 'tag_jaccard_similarity',
            'vault_path_distance', 'shortest_path_length', 'common_neighbors_count',
            'adamic_adar_score', 'word_count_ratio', 'creation_time_diff_days',
            'quality_stage_compatibility', 'source_centrality', 'target_centrality',
            'clustering_coefficient', 'content_length_ratio', 'title_similarity'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ensure boolean columns
        boolean_columns = ['semantic_cluster_match', 'link_exists', 'same_folder']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        return df

    def validate_inputs(self, notes_data: dict[str, NoteData],
                       link_graph: nx.DiGraph) -> ValidationResult:
        """Validate input parameters and data quality.
        
        Args:
            notes_data: Dictionary of note data
            link_graph: NetworkX graph
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(valid=True)

        # Check minimum notes requirement
        if len(notes_data) < 5:
            result.valid = False
            result.errors.append(f"Insufficient notes: {len(notes_data)} < 5 required")

        # Check for embeddings
        notes_with_embeddings = sum(1 for note in notes_data.values() if note.embedding is not None)
        if notes_with_embeddings == 0:
            result.warnings.append("No notes have embeddings - semantic similarity will be 0")

        # Check graph validity
        if not isinstance(link_graph, nx.DiGraph):
            result.valid = False
            result.errors.append("Invalid graph type")

        # Check for positive pairs
        positive_pairs = len(link_graph.edges())
        if positive_pairs == 0:
            result.warnings.append("No positive pairs found in graph")

        result.notes_processed = len(notes_data)
        result.links_extracted = positive_pairs

        return result

    def get_feature_descriptions(self) -> dict[str, str]:
        """Get descriptions of all generated features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'note_a_path': 'Path to the first note in the pair',
            'note_b_path': 'Path to the second note in the pair',
            'cosine_similarity': 'Cosine similarity between note embeddings',
            'semantic_cluster_match': 'Whether notes belong to the same semantic cluster',
            'tag_overlap_count': 'Number of tags shared between the notes',
            'tag_jaccard_similarity': 'Jaccard similarity of tag sets',
            'vault_path_distance': 'Directory distance between note paths',
            'shortest_path_length': 'Shortest path length in the link graph',
            'common_neighbors_count': 'Number of common neighbor notes in the graph',
            'adamic_adar_score': 'Adamic-Adar score for link prediction',
            'word_count_ratio': 'Ratio of word counts (smaller/larger)',
            'creation_time_diff_days': 'Difference in creation dates (days)',
            'quality_stage_compatibility': 'Whether notes have compatible quality stages',
            'source_centrality': 'Combined in/out degree centrality of first note',
            'target_centrality': 'Combined in/out degree centrality of second note',
            'clustering_coefficient': 'Average clustering coefficient of the pair',
            'link_exists': 'Whether a link exists between the notes (target variable)',
            'same_folder': 'Whether notes are in the same folder',
            'content_length_ratio': 'Ratio of content lengths (smaller/larger)',
            'title_similarity': 'Jaccard similarity of title words'
        }

    def get_processing_stats(self) -> ProcessingStats | None:
        """Get processing statistics from the last generation run.
        
        Returns:
            ProcessingStats object or None if no processing has occurred
        """
        return self._processing_stats
