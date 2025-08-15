"""
Feature engineer for comprehensive dataset generation.

This module integrates all analyzers to provide a unified interface for
extracting enhanced features from notes and note pairs with comprehensive
error handling and graceful degradation.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.logging import setup_logging

from .analyzers.content_analyzer import ContentAnalyzer, ContentFeatures
from .analyzers.graph_analyzer import AdvancedCentralityMetrics, GraphAnalyzer
from .analyzers.semantic_analyzer import SemanticAnalyzer
from .analyzers.topic_modeler import TopicModeler, TopicModelResult
from .error_handling import (
    ComponentType,
    ErrorSeverity,
    FallbackValues,
    create_minimal_features,
    ensure_feature_quality,
    log_system_health,
    with_error_handling,
)
from .models.data_models import NoteData, NoteFeatures, PairFeatures

logger = setup_logging(__name__)


@dataclass
class EnhancedFeatures:
    """Container for all enhanced features."""
    # Semantic features
    semantic_similarity: float = 0.0
    tfidf_similarity: float = 0.0
    combined_similarity: float = 0.0

    # Content features
    content_features: ContentFeatures | None = None

    # Topic features
    dominant_topic_id: int = -1
    dominant_topic_probability: float = 0.0
    topic_probabilities: list[float] = field(default_factory=list)
    topic_label: str = ""

    # Graph features
    centrality_metrics: AdvancedCentralityMetrics | None = None

    # TF-IDF features
    top_tfidf_terms: list[tuple[str, float]] = field(default_factory=list)
    tfidf_vocabulary_richness: float = 0.0
    avg_tfidf_score: float = 0.0


class FeatureEngineer:
    """Unified feature engineering for dataset generation."""

    def __init__(self, vector_encoder: VectorEncoder):
        """Initialize the feature engineer.
        
        Args:
            vector_encoder: VectorEncoder service for embeddings
        """
        self.vector_encoder = vector_encoder

        # Initialize analyzers
        self.semantic_analyzer = SemanticAnalyzer(vector_encoder)
        self.content_analyzer = ContentAnalyzer(use_spacy=True, use_textstat=True)
        self.topic_modeler = TopicModeler(method="auto", min_topic_size=5)
        self.graph_analyzer = GraphAnalyzer()

        # Fitted state
        self._semantic_fitted = False
        self._topic_fitted = False
        self._topic_result = None
        self._topic_predictions_cache = {}  # Cache for pre-computed topic predictions

    def fit_analyzers(self, notes_data: list[NoteData]) -> dict[str, Any]:
        """Fit all analyzers on the provided notes data with comprehensive error handling.
        
        Args:
            notes_data: List of NoteData objects
            
        Returns:
            Dictionary with fitting results and metadata
        """
        logger.info(f"Fitting analyzers on {len(notes_data)} notes")

        # Check system health before starting
        health_info = log_system_health()

        results = {
            'semantic_fitted': False,
            'topic_fitted': False,
            'notes_processed': len(notes_data),
            'errors': [],
            'system_health': health_info
        }

        # Extract texts for fitting
        texts = []
        embeddings_list = []

        for note_data in notes_data:
            if note_data.content and note_data.content.strip():
                texts.append(note_data.content)
                if note_data.embedding is not None:
                    embeddings_list.append(note_data.embedding)

        if not texts:
            logger.warning("No valid texts found for analyzer fitting")
            return results

        # Fit semantic analyzer with error handling
        semantic_results = self._fit_semantic_analyzer_safe(texts)
        if semantic_results:
            self._semantic_fitted = True
            results['semantic_fitted'] = True
            results['semantic_results'] = semantic_results
            logger.info(f"Semantic analyzer fitted: {semantic_results.get('tfidf_features', 0)} TF-IDF features")
        else:
            results['errors'].append("Semantic analyzer fitting failed - using fallback mode")

        # Fit topic modeler with error handling
        embeddings_array = np.array(embeddings_list) if embeddings_list else None
        topic_result = self._fit_topic_modeler_safe(texts, embeddings_array)
        if topic_result and topic_result.topic_count > 0:
            self._topic_fitted = True
            self._topic_result = topic_result
            results['topic_fitted'] = True
            results['topic_count'] = topic_result.topic_count
            logger.info(f"Topic modeler fitted: {topic_result.topic_count} topics")

            # Pre-compute and cache topic predictions for all notes
            logger.info("Pre-computing topic predictions for all notes to optimize pair generation")
            self._precompute_topic_predictions(notes_data)
            logger.info(f"Topic predictions cached for {len(self._topic_predictions_cache)} notes")
        else:
            results['errors'].append("Topic modeler fitting failed - using fallback mode")

        return results

    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="semantic_analyzer_fitting",
        fallback_value=None,
        severity=ErrorSeverity.HIGH,
        recovery_action="Check if sentence-transformers and sklearn are installed"
    )
    def _fit_semantic_analyzer_safe(self, texts: list[str]) -> dict[str, Any] | None:
        """Safely fit the semantic analyzer with error handling."""
        return self.semantic_analyzer.fit_and_transform(texts)

    @with_error_handling(
        component=ComponentType.TOPIC_MODELER,
        feature_name="topic_modeler_fitting",
        fallback_value=None,
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Check if BERTopic or scikit-learn are installed"
    )
    def _fit_topic_modeler_safe(self, texts: list[str], embeddings: np.ndarray | None) -> TopicModelResult | None:
        """Safely fit the topic modeler with error handling."""
        return self.topic_modeler.fit_topics(texts, embeddings)

    def extract_note_features(self, note_data: NoteData) -> EnhancedFeatures:
        """Extract comprehensive features for a single note with robust error handling.
        
        Args:
            note_data: NoteData object
            
        Returns:
            EnhancedFeatures object with all extracted features
        """
        features = EnhancedFeatures()

        if not note_data.content or not note_data.content.strip():
            logger.debug(f"No content available for {note_data.path}")
            return features

        # Content analysis with error handling
        content_features = self._extract_content_features_safe(note_data.content, note_data.path)
        if content_features:
            features.content_features = content_features

        # TF-IDF features with error handling
        tfidf_features = self._extract_tfidf_features_safe(note_data.content, note_data.path)
        features.avg_tfidf_score = tfidf_features.get('avg_tfidf_score', FallbackValues.AVG_TFIDF_SCORE)
        features.tfidf_vocabulary_richness = tfidf_features.get('tfidf_vocabulary_richness', FallbackValues.TFIDF_VOCABULARY_RICHNESS)
        features.top_tfidf_terms = tfidf_features.get('top_tfidf_terms', FallbackValues.TOP_TFIDF_TERMS)

        # Topic modeling features with error handling
        topic_features = self._extract_topic_features_safe(note_data.content, note_data.embedding, note_data.path)
        features.dominant_topic_id = topic_features.get('dominant_topic_id', FallbackValues.DOMINANT_TOPIC_ID)
        features.dominant_topic_probability = topic_features.get('dominant_topic_probability', FallbackValues.DOMINANT_TOPIC_PROBABILITY)
        features.topic_probabilities = topic_features.get('topic_probabilities', FallbackValues.TOPIC_PROBABILITIES)
        features.topic_label = topic_features.get('topic_label', FallbackValues.TOPIC_LABEL)

        # Validate and ensure feature quality
        feature_dict = {
            'avg_tfidf_score': features.avg_tfidf_score,
            'tfidf_vocabulary_richness': features.tfidf_vocabulary_richness,
            'dominant_topic_probability': features.dominant_topic_probability,
            'topic_similarity': features.topic_probabilities
        }

        validated_features = ensure_feature_quality(feature_dict, ComponentType.FEATURE_ENGINEER)

        # Update features with validated values
        features.avg_tfidf_score = validated_features.get('avg_tfidf_score', features.avg_tfidf_score)
        features.tfidf_vocabulary_richness = validated_features.get('tfidf_vocabulary_richness', features.tfidf_vocabulary_richness)
        features.dominant_topic_probability = validated_features.get('dominant_topic_probability', features.dominant_topic_probability)

        return features

    def _precompute_topic_predictions(self, notes_data: list[NoteData]) -> None:
        """Pre-compute topic predictions for all notes to avoid repeated BERTopic calls.
        
        Args:
            notes_data: List of NoteData objects
        """
        if not self._topic_fitted or not self._topic_result:
            return

        try:
            # Batch predict topics for all notes at once
            all_texts = []
            all_embeddings = []
            note_paths = []

            for note_data in notes_data:
                if note_data.content and note_data.content.strip():
                    all_texts.append(note_data.content)
                    note_paths.append(note_data.path)
                    if note_data.embedding is not None:
                        all_embeddings.append(note_data.embedding)
                    else:
                        all_embeddings.append(None)

            if not all_texts:
                return

            # Convert embeddings to array (handle None values)
            embeddings_array = None
            if any(emb is not None for emb in all_embeddings):
                valid_embeddings = []
                valid_indices = []
                for i, emb in enumerate(all_embeddings):
                    if emb is not None:
                        valid_embeddings.append(emb)
                        valid_indices.append(i)

                if valid_embeddings:
                    embeddings_array = np.array(valid_embeddings)

            # Batch predict topics for all texts at once
            logger.debug(f"Batch predicting topics for {len(all_texts)} documents")
            topic_prediction = self.topic_modeler.predict_topics(all_texts, embeddings_array)

            # Cache the predictions by note path
            for i, note_path in enumerate(note_paths):
                if i < len(topic_prediction.topic_assignments):
                    dominant_topic_id = topic_prediction.topic_assignments[i]
                    topic_probabilities = topic_prediction.topic_probabilities[i] if topic_prediction.topic_probabilities else []
                    dominant_topic_probability = max(topic_probabilities) if topic_probabilities else 0.0
                    topic_label = self._topic_result.topic_labels.get(dominant_topic_id, f"Topic {dominant_topic_id}")

                    self._topic_predictions_cache[note_path] = {
                        'dominant_topic_id': dominant_topic_id,
                        'dominant_topic_probability': dominant_topic_probability,
                        'topic_probabilities': topic_probabilities,
                        'topic_label': topic_label
                    }

            logger.info(f"Successfully cached topic predictions for {len(self._topic_predictions_cache)} notes")

        except Exception as e:
            logger.warning(f"Failed to pre-compute topic predictions: {e}")
            # Clear the cache on error
            self._topic_predictions_cache.clear()

    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="content_analysis",
        fallback_value=None,
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Check if spaCy and textstat are installed"
    )
    def _extract_content_features_safe(self, content: str, note_path: str) -> ContentFeatures | None:
        """Safely extract content features with error handling."""
        return self.content_analyzer.analyze_content(content)

    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="tfidf_features",
        fallback_value={},
        severity=ErrorSeverity.LOW,
        recovery_action="TF-IDF features will use default values"
    )
    def _extract_tfidf_features_safe(self, content: str, note_path: str) -> dict[str, Any]:
        """Safely extract TF-IDF features with error handling."""
        if not self._semantic_fitted:
            return {}

        # Get TF-IDF features for this document
        tfidf_matrix = self.semantic_analyzer.compute_tfidf_features([content])

        if tfidf_matrix.shape[0] == 0:
            return {}

        # Calculate TF-IDF statistics
        tfidf_vector = tfidf_matrix[0]
        avg_tfidf_score = float(tfidf_vector.mean()) if tfidf_vector.nnz > 0 else 0.0

        # Calculate vocabulary richness
        total_features = tfidf_matrix.shape[1]
        non_zero_features = tfidf_vector.nnz
        tfidf_vocabulary_richness = non_zero_features / total_features if total_features > 0 else 0.0

        return {
            'avg_tfidf_score': avg_tfidf_score,
            'tfidf_vocabulary_richness': tfidf_vocabulary_richness,
            'top_tfidf_terms': []  # Could be implemented later
        }

    @with_error_handling(
        component=ComponentType.TOPIC_MODELER,
        feature_name="topic_features",
        fallback_value={},
        severity=ErrorSeverity.LOW,
        recovery_action="Topic features will use default values"
    )
    def _extract_topic_features_safe(self, content: str, embedding: np.ndarray | None, note_path: str) -> dict[str, Any]:
        """Safely extract topic modeling features with error handling."""
        if not self._topic_fitted or not self._topic_result:
            return {}

        # First check cache for pre-computed predictions
        if note_path in self._topic_predictions_cache:
            logger.debug(f"Using cached topic prediction for {note_path}")
            return self._topic_predictions_cache[note_path]

        # Fallback to individual prediction if not in cache
        logger.debug(f"Computing topic prediction for {note_path} (not in cache)")

        # Predict topic for this document
        embeddings = np.array([embedding]) if embedding is not None else None
        topic_prediction = self.topic_modeler.predict_topics([content], embeddings)

        if not topic_prediction.topic_assignments:
            return {}

        dominant_topic_id = topic_prediction.topic_assignments[0]
        topic_probabilities = topic_prediction.topic_probabilities[0] if topic_prediction.topic_probabilities else []
        dominant_topic_probability = max(topic_probabilities) if topic_probabilities else 0.0

        # Get topic label
        topic_label = self._topic_result.topic_labels.get(dominant_topic_id, f"Topic {dominant_topic_id}")

        result = {
            'dominant_topic_id': dominant_topic_id,
            'dominant_topic_probability': dominant_topic_probability,
            'topic_probabilities': topic_probabilities,
            'topic_label': topic_label
        }

        # Cache the result for future use
        self._topic_predictions_cache[note_path] = result

        return result

    def extract_pair_features(self, note_a: NoteData, note_b: NoteData) -> dict[str, float]:
        """Extract enhanced similarity features for a pair of notes with robust error handling.
        
        Args:
            note_a: First note data
            note_b: Second note data
            
        Returns:
            Dictionary with enhanced similarity features
        """
        # Initialize with fallback values
        features = create_minimal_features(ComponentType.FEATURE_ENGINEER, "pair")

        if not (note_a.content and note_b.content):
            logger.debug(f"Missing content for pair: {note_a.path} <-> {note_b.path}")
            return features

        # Semantic and TF-IDF similarity with error handling
        semantic_features = self._extract_semantic_pair_features_safe(note_a.content, note_b.content)
        features.update(semantic_features)

        # Topic similarity with error handling
        topic_features = self._extract_topic_pair_features_safe(note_a, note_b)
        features.update(topic_features)

        # Content-based similarity with error handling
        content_features = self._extract_content_pair_features_safe(note_a.content, note_b.content)
        features.update(content_features)

        # Validate and ensure feature quality
        validated_features = ensure_feature_quality(features, ComponentType.FEATURE_ENGINEER)

        return validated_features

    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="semantic_pair_features",
        fallback_value={},
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Semantic similarity will use default values"
    )
    def _extract_semantic_pair_features_safe(self, content_a: str, content_b: str) -> dict[str, float]:
        """Safely extract semantic pair features with error handling."""
        if not self._semantic_fitted:
            return {}

        return self.semantic_analyzer.extract_pair_features(content_a, content_b)

    @with_error_handling(
        component=ComponentType.TOPIC_MODELER,
        feature_name="topic_pair_features",
        fallback_value={},
        severity=ErrorSeverity.LOW,
        recovery_action="Topic similarity will use default values"
    )
    def _extract_topic_pair_features_safe(self, note_a: NoteData, note_b: NoteData) -> dict[str, float]:
        """Safely extract topic pair features with error handling using cached predictions."""
        if not self._topic_fitted or not self._topic_result:
            return {}

        # Get cached topic predictions for both notes
        topic_features_a = None
        topic_features_b = None

        # Try to get cached predictions first
        if note_a.path in self._topic_predictions_cache:
            topic_features_a = self._topic_predictions_cache[note_a.path]
        else:
            # Fallback to individual prediction
            topic_features_a = self._extract_topic_features_safe(note_a.content, note_a.embedding, note_a.path)

        if note_b.path in self._topic_predictions_cache:
            topic_features_b = self._topic_predictions_cache[note_b.path]
        else:
            # Fallback to individual prediction
            topic_features_b = self._extract_topic_features_safe(note_b.content, note_b.embedding, note_b.path)

        if not topic_features_a or not topic_features_b:
            return {}

        # Check if same dominant topic
        topic_a = topic_features_a.get('dominant_topic_id', -1)
        topic_b = topic_features_b.get('dominant_topic_id', -1)
        same_dominant_topic = (topic_a == topic_b and topic_a != -1)

        # Calculate topic probability similarity (cosine similarity of probability vectors)
        topic_similarity = 0.0
        probs_a = np.array(topic_features_a.get('topic_probabilities', []))
        probs_b = np.array(topic_features_b.get('topic_probabilities', []))

        if len(probs_a) == len(probs_b) and len(probs_a) > 0:
            norm_a = np.linalg.norm(probs_a)
            norm_b = np.linalg.norm(probs_b)

            if norm_a > 0 and norm_b > 0:
                topic_similarity = float(np.dot(probs_a, probs_b) / (norm_a * norm_b))

        return {
            'topic_similarity': topic_similarity,
            'same_dominant_topic': same_dominant_topic
        }

    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="content_pair_features",
        fallback_value={},
        severity=ErrorSeverity.LOW,
        recovery_action="Content similarity will use default values"
    )
    def _extract_content_pair_features_safe(self, content_a: str, content_b: str) -> dict[str, float]:
        """Safely extract content pair features with error handling."""
        content_features_a = self.content_analyzer.analyze_content(content_a)
        content_features_b = self.content_analyzer.analyze_content(content_b)

        # Simple content similarity based on multiple factors
        similarity_factors = []

        # Sentiment similarity
        if abs(content_features_a.sentiment_score - content_features_b.sentiment_score) <= 2.0:
            sentiment_sim = 1.0 - abs(content_features_a.sentiment_score - content_features_b.sentiment_score) / 2.0
            similarity_factors.append(sentiment_sim)

        # Content type similarity
        if content_features_a.content_type == content_features_b.content_type:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)

        # Complexity similarity
        if abs(content_features_a.complexity_score - content_features_b.complexity_score) <= 1.0:
            complexity_sim = 1.0 - abs(content_features_a.complexity_score - content_features_b.complexity_score)
            similarity_factors.append(complexity_sim)

        # Average similarity
        content_similarity = sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

        return {'content_similarity': content_similarity}

    def update_note_features_with_enhanced(self, note_features: NoteFeatures,
                                         enhanced_features: EnhancedFeatures) -> NoteFeatures:
        """Update NoteFeatures object with enhanced features.
        
        Args:
            note_features: Existing NoteFeatures object
            enhanced_features: EnhancedFeatures to integrate
            
        Returns:
            Updated NoteFeatures object
        """
        try:
            # Content analysis features
            if enhanced_features.content_features:
                cf = enhanced_features.content_features
                note_features.sentiment_score = cf.sentiment_score
                note_features.sentiment_label = cf.sentiment_label
                note_features.readability_score = cf.readability_score
                note_features.readability_grade = cf.readability_grade
                note_features.complexity_score = cf.complexity_score
                note_features.vocabulary_richness = cf.vocabulary_richness
                note_features.content_type = cf.content_type

                # Convert named entities to JSON
                note_features.named_entities_json = json.dumps(cf.named_entities)
                note_features.entity_types_json = json.dumps(cf.entity_types)

                # Update structure features
                note_features.heading_count = cf.heading_count
                note_features.max_heading_depth = cf.max_heading_depth

                # Update density features
                note_features.technical_term_density = cf.technical_density
                note_features.concept_density_score = cf.concept_density

            # TF-IDF features
            note_features.tfidf_vocabulary_richness = enhanced_features.tfidf_vocabulary_richness
            note_features.avg_tfidf_score = enhanced_features.avg_tfidf_score

            if enhanced_features.top_tfidf_terms:
                note_features.top_tfidf_terms = json.dumps([
                    {"term": term, "score": float(score)}
                    for term, score in enhanced_features.top_tfidf_terms
                ])

            # Topic modeling features
            note_features.dominant_topic_id = enhanced_features.dominant_topic_id
            note_features.dominant_topic_probability = enhanced_features.dominant_topic_probability
            note_features.topic_label = enhanced_features.topic_label

            if enhanced_features.topic_probabilities:
                note_features.topic_probabilities_json = json.dumps(enhanced_features.topic_probabilities)

            # Graph features (if available)
            if enhanced_features.centrality_metrics:
                cm = enhanced_features.centrality_metrics
                note_features.pagerank_score = cm.pagerank
                note_features.betweenness_centrality = cm.betweenness_centrality
                note_features.closeness_centrality = cm.closeness_centrality
                note_features.clustering_coefficient = cm.clustering_coefficient

        except Exception as e:
            logger.error(f"Error updating note features: {e}")

        return note_features

    def update_pair_features_with_enhanced(self, pair_features: PairFeatures,
                                         enhanced_features: dict[str, float]) -> PairFeatures:
        """Update PairFeatures object with enhanced features.
        
        Args:
            pair_features: Existing PairFeatures object
            enhanced_features: Enhanced similarity features
            
        Returns:
            Updated PairFeatures object
        """
        try:
            # Update similarity features
            pair_features.cosine_similarity = enhanced_features.get('semantic_similarity', pair_features.cosine_similarity)
            pair_features.tfidf_similarity = enhanced_features.get('tfidf_similarity', pair_features.tfidf_similarity)
            pair_features.combined_similarity = enhanced_features.get('combined_similarity', pair_features.combined_similarity)

            # Add new topic-based features
            pair_features.topic_similarity = enhanced_features.get('topic_similarity', 0.0)
            pair_features.same_dominant_topic = enhanced_features.get('same_dominant_topic', False)

            # Calculate topic coherence average (placeholder - would need actual coherence scores)
            pair_features.topic_coherence_avg = 0.0  # This would be calculated from individual note coherence scores

        except Exception as e:
            logger.error(f"Error updating pair features: {e}")

        return pair_features

    def get_analyzer_status(self) -> dict[str, Any]:
        """Get status of all analyzers.
        
        Returns:
            Dictionary with analyzer status information
        """
        return {
            'semantic_analyzer': {
                'fitted': self._semantic_fitted,
                'stats': self.semantic_analyzer.get_vocabulary_stats() if self._semantic_fitted else {}
            },
            'content_analyzer': {
                'spacy_available': self.content_analyzer._spacy_available,
                'textstat_available': self.content_analyzer._textstat_available
            },
            'topic_modeler': {
                'fitted': self._topic_fitted,
                'model_info': self.topic_modeler.get_model_info(),
                'topic_count': self._topic_result.topic_count if self._topic_result else 0
            },
            'graph_analyzer': {
                'available': True  # Graph analyzer doesn't require fitting
            }
        }
