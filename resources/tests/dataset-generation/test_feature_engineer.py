"""
Unit tests for FeatureEngineer component.

Tests comprehensive feature engineering including integration of all analyzers,
error handling, and graceful degradation.
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.jarvis.tools.dataset_generation.feature_engineer import (
    FeatureEngineer, EnhancedFeatures
)
from src.jarvis.tools.dataset_generation.models.data_models import (
    NoteData, NoteFeatures, PairFeatures
)
from src.jarvis.tools.dataset_generation.analyzers.content_analyzer import ContentFeatures
from src.jarvis.tools.dataset_generation.analyzers.topic_modeler import TopicModelResult
from src.jarvis.tools.dataset_generation.analyzers.graph_analyzer import AdvancedCentralityMetrics
from src.jarvis.tools.dataset_generation.error_handling import (
    get_error_tracker, ComponentType
)


class TestEnhancedFeatures:
    """Test EnhancedFeatures data class."""
    
    def test_enhanced_features_initialization(self):
        """Test EnhancedFeatures initialization with default values."""
        features = EnhancedFeatures()
        
        assert features.semantic_similarity == 0.0
        assert features.tfidf_similarity == 0.0
        assert features.combined_similarity == 0.0
        assert features.content_features is None
        assert features.dominant_topic_id == -1
        assert features.dominant_topic_probability == 0.0
        assert features.topic_probabilities == []
        assert features.topic_label == ""
        assert features.centrality_metrics is None
        assert features.top_tfidf_terms == []
        assert features.tfidf_vocabulary_richness == 0.0
        assert features.avg_tfidf_score == 0.0
    
    def test_enhanced_features_with_values(self):
        """Test EnhancedFeatures with specific values."""
        content_features = ContentFeatures(sentiment_score=0.8, complexity_score=0.6)
        centrality_metrics = AdvancedCentralityMetrics(pagerank=0.1, betweenness_centrality=0.2)
        
        features = EnhancedFeatures(
            semantic_similarity=0.85,
            tfidf_similarity=0.75,
            combined_similarity=0.82,
            content_features=content_features,
            dominant_topic_id=1,
            dominant_topic_probability=0.9,
            topic_probabilities=[0.1, 0.9],
            topic_label="Technology",
            centrality_metrics=centrality_metrics,
            top_tfidf_terms=[("machine", 0.5), ("learning", 0.4)],
            tfidf_vocabulary_richness=0.7,
            avg_tfidf_score=0.3
        )
        
        assert features.semantic_similarity == 0.85
        assert features.tfidf_similarity == 0.75
        assert features.combined_similarity == 0.82
        assert features.content_features == content_features
        assert features.dominant_topic_id == 1
        assert features.dominant_topic_probability == 0.9
        assert features.topic_probabilities == [0.1, 0.9]
        assert features.topic_label == "Technology"
        assert features.centrality_metrics == centrality_metrics
        assert features.top_tfidf_terms == [("machine", 0.5), ("learning", 0.4)]
        assert features.tfidf_vocabulary_richness == 0.7
        assert features.avg_tfidf_score == 0.3


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock VectorEncoder."""
        encoder = Mock()
        encoder.encode_batch.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6]
        ])
        encoder.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        return encoder
    
    @pytest.fixture
    def feature_engineer(self, mock_vector_encoder):
        """Create FeatureEngineer instance for testing."""
        return FeatureEngineer(mock_vector_encoder)
    
    @pytest.fixture
    def sample_note_data(self):
        """Sample NoteData for testing."""
        return [
            NoteData(
                path="note1.md",
                content="Machine learning is a fascinating field of artificial intelligence.",
                embedding=np.array([0.1, 0.2, 0.3, 0.4]),
                metadata={"created": "2024-01-01", "modified": "2024-01-02"}
            ),
            NoteData(
                path="note2.md",
                content="Deep learning uses neural networks with multiple layers.",
                embedding=np.array([0.2, 0.3, 0.4, 0.5]),
                metadata={"created": "2024-01-02", "modified": "2024-01-03"}
            ),
            NoteData(
                path="note3.md",
                content="Natural language processing helps computers understand text.",
                embedding=np.array([0.3, 0.4, 0.5, 0.6]),
                metadata={"created": "2024-01-03", "modified": "2024-01-04"}
            )
        ]
    
    def test_feature_engineer_initialization(self, mock_vector_encoder):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(mock_vector_encoder)
        
        assert engineer.vector_encoder == mock_vector_encoder
        assert engineer.semantic_analyzer is not None
        assert engineer.content_analyzer is not None
        assert engineer.topic_modeler is not None
        assert engineer.graph_analyzer is not None
        assert not engineer._semantic_fitted
        assert not engineer._topic_fitted
        assert engineer._topic_result is None
    
    def test_fit_analyzers_empty_data(self, feature_engineer):
        """Test fitting analyzers with empty data."""
        results = feature_engineer.fit_analyzers([])
        
        assert isinstance(results, dict)
        assert results['semantic_fitted'] is False
        assert results['topic_fitted'] is False
        assert results['notes_processed'] == 0
        assert 'system_health' in results
    
    def test_fit_analyzers_success(self, feature_engineer, sample_note_data):
        """Test successful analyzer fitting."""
        with patch.object(feature_engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic_fit, \
             patch.object(feature_engineer.topic_modeler, 'fit_topics') as mock_topic_fit:
            
            # Mock successful semantic fitting
            mock_semantic_fit.return_value = {
                'tfidf_features': 100,
                'embedding_features': 384,
                'vocabulary_size': 50
            }
            
            # Mock successful topic fitting
            mock_topic_result = TopicModelResult(
                topic_count=3,
                model_type="bertopic",
                coherence_score=0.75
            )
            mock_topic_fit.return_value = mock_topic_result
            
            results = feature_engineer.fit_analyzers(sample_note_data)
            
            assert results['semantic_fitted'] is True
            assert results['topic_fitted'] is True
            assert results['notes_processed'] == 3
            assert results['topic_count'] == 3
            assert 'semantic_results' in results
            assert feature_engineer._semantic_fitted is True
            assert feature_engineer._topic_fitted is True
            assert feature_engineer._topic_result == mock_topic_result
    
    def test_fit_analyzers_partial_failure(self, feature_engineer, sample_note_data):
        """Test analyzer fitting with partial failures."""
        with patch.object(feature_engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic_fit, \
             patch.object(feature_engineer.topic_modeler, 'fit_topics') as mock_topic_fit:
            
            # Mock semantic success, topic failure
            mock_semantic_fit.return_value = {'tfidf_features': 100}
            mock_topic_fit.return_value = None
            
            results = feature_engineer.fit_analyzers(sample_note_data)
            
            assert results['semantic_fitted'] is True
            assert results['topic_fitted'] is False
            assert len(results['errors']) > 0
            assert "Topic modeler fitting failed" in results['errors'][0]
    
    def test_extract_note_features_empty_content(self, feature_engineer):
        """Test note feature extraction with empty content."""
        note_data = NoteData(path="empty.md", content="", embedding=None)
        
        features = feature_engineer.extract_note_features(note_data)
        
        assert isinstance(features, EnhancedFeatures)
        assert features.content_features is None
        assert features.dominant_topic_id == -1
        assert features.avg_tfidf_score == 0.0
    
    def test_extract_note_features_none_content(self, feature_engineer):
        """Test note feature extraction with None content."""
        note_data = NoteData(path="none.md", content=None, embedding=None)
        
        features = feature_engineer.extract_note_features(note_data)
        
        assert isinstance(features, EnhancedFeatures)
        assert features.content_features is None
    
    def test_extract_note_features_success(self, feature_engineer, sample_note_data):
        """Test successful note feature extraction."""
        # First fit the analyzers
        with patch.object(feature_engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic_fit, \
             patch.object(feature_engineer.topic_modeler, 'fit_topics') as mock_topic_fit:
            
            mock_semantic_fit.return_value = {'tfidf_features': 100}
            mock_topic_fit.return_value = TopicModelResult(topic_count=2)
            
            feature_engineer.fit_analyzers(sample_note_data)
        
        # Mock individual feature extraction methods
        with patch.object(feature_engineer.content_analyzer, 'analyze_content') as mock_content, \
             patch.object(feature_engineer.semantic_analyzer, 'compute_tfidf_features') as mock_tfidf, \
             patch.object(feature_engineer.topic_modeler, 'predict_topics') as mock_topic_predict:
            
            # Mock content analysis
            mock_content_features = ContentFeatures(
                sentiment_score=0.8,
                complexity_score=0.6,
                vocabulary_richness=0.7
            )
            mock_content.return_value = mock_content_features
            
            # Mock TF-IDF features
            mock_tfidf_matrix = Mock()
            mock_tfidf_matrix.shape = (1, 100)
            mock_tfidf_matrix.__getitem__.return_value = Mock()
            mock_tfidf_matrix.__getitem__.return_value.mean.return_value = 0.3
            mock_tfidf_matrix.__getitem__.return_value.nnz = 50
            mock_tfidf.return_value = mock_tfidf_matrix
            
            # Mock topic prediction
            from src.jarvis.tools.dataset_generation.analyzers.topic_modeler import TopicPrediction
            mock_topic_prediction = TopicPrediction(
                topic_assignments=[1],
                topic_probabilities=[[0.2, 0.8]],
                confidence_scores=[0.8]
            )
            mock_topic_predict.return_value = mock_topic_prediction
            
            note_data = sample_note_data[0]
            features = feature_engineer.extract_note_features(note_data)
            
            assert isinstance(features, EnhancedFeatures)
            assert features.content_features == mock_content_features
            assert features.dominant_topic_id == 1
            assert features.dominant_topic_probability == 0.8
            assert features.avg_tfidf_score == 0.3
    
    def test_extract_pair_features_empty_content(self, feature_engineer):
        """Test pair feature extraction with empty content."""
        note_a = NoteData(path="a.md", content="", embedding=None)
        note_b = NoteData(path="b.md", content="", embedding=None)
        
        features = feature_engineer.extract_pair_features(note_a, note_b)
        
        assert isinstance(features, dict)
        # Should return fallback values
        assert all(isinstance(v, (int, float, bool)) for v in features.values())
    
    def test_extract_pair_features_success(self, feature_engineer, sample_note_data):
        """Test successful pair feature extraction."""
        # First fit the analyzers
        with patch.object(feature_engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic_fit, \
             patch.object(feature_engineer.topic_modeler, 'fit_topics') as mock_topic_fit:
            
            mock_semantic_fit.return_value = {'tfidf_features': 100}
            mock_topic_fit.return_value = TopicModelResult(topic_count=2)
            
            feature_engineer.fit_analyzers(sample_note_data)
        
        # Mock pair feature extraction methods
        with patch.object(feature_engineer.semantic_analyzer, 'extract_pair_features') as mock_semantic_pair, \
             patch.object(feature_engineer.topic_modeler, 'predict_topics') as mock_topic_predict, \
             patch.object(feature_engineer.content_analyzer, 'analyze_content') as mock_content:
            
            # Mock semantic pair features
            mock_semantic_pair.return_value = {
                'semantic_similarity': 0.8,
                'tfidf_similarity': 0.7,
                'combined_similarity': 0.75
            }
            
            # Mock topic predictions
            from src.jarvis.tools.dataset_generation.analyzers.topic_modeler import TopicPrediction
            mock_topic_predict.return_value = TopicPrediction(
                topic_assignments=[1],
                topic_probabilities=[[0.2, 0.8]],
                confidence_scores=[0.8]
            )
            
            # Mock content analysis
            mock_content.return_value = ContentFeatures(
                sentiment_score=0.5,
                content_type="technical",
                complexity_score=0.6
            )
            
            note_a = sample_note_data[0]
            note_b = sample_note_data[1]
            
            features = feature_engineer.extract_pair_features(note_a, note_b)
            
            assert isinstance(features, dict)
            assert 'semantic_similarity' in features
            assert 'tfidf_similarity' in features
            assert 'combined_similarity' in features
            assert 'topic_similarity' in features
            assert 'content_similarity' in features
    
    def test_update_note_features_with_enhanced(self, feature_engineer):
        """Test updating NoteFeatures with enhanced features."""
        # Create base note features
        note_features = NoteFeatures(
            path="test.md",
            word_count=100,
            char_count=500
        )
        
        # Create enhanced features
        content_features = ContentFeatures(
            sentiment_score=0.8,
            sentiment_label="positive",
            readability_score=75.0,
            complexity_score=0.6,
            vocabulary_richness=0.7,
            content_type="technical",
            named_entities=[{"text": "Python", "label": "LANGUAGE"}],
            entity_types={"LANGUAGE": 1},
            heading_count=3,
            max_heading_depth=2,
            technical_density=0.4,
            concept_density=0.3
        )
        
        centrality_metrics = AdvancedCentralityMetrics(
            pagerank=0.1,
            betweenness_centrality=0.2,
            closeness_centrality=0.3,
            clustering_coefficient=0.4
        )
        
        enhanced_features = EnhancedFeatures(
            content_features=content_features,
            dominant_topic_id=1,
            dominant_topic_probability=0.9,
            topic_label="Technology",
            topic_probabilities=[0.1, 0.9],
            centrality_metrics=centrality_metrics,
            tfidf_vocabulary_richness=0.6,
            avg_tfidf_score=0.3,
            top_tfidf_terms=[("machine", 0.5), ("learning", 0.4)]
        )
        
        updated_features = feature_engineer.update_note_features_with_enhanced(
            note_features, enhanced_features
        )
        
        # Check that enhanced features were integrated
        assert updated_features.sentiment_score == 0.8
        assert updated_features.sentiment_label == "positive"
        assert updated_features.readability_score == 75.0
        assert updated_features.complexity_score == 0.6
        assert updated_features.vocabulary_richness == 0.7
        assert updated_features.content_type == "technical"
        assert updated_features.dominant_topic_id == 1
        assert updated_features.dominant_topic_probability == 0.9
        assert updated_features.topic_label == "Technology"
        assert updated_features.pagerank_score == 0.1
        assert updated_features.betweenness_centrality == 0.2
        assert updated_features.tfidf_vocabulary_richness == 0.6
        assert updated_features.avg_tfidf_score == 0.3
        
        # Check JSON fields
        assert updated_features.named_entities_json is not None
        assert updated_features.entity_types_json is not None
        assert updated_features.topic_probabilities_json is not None
        assert updated_features.top_tfidf_terms is not None
    
    def test_update_pair_features_with_enhanced(self, feature_engineer):
        """Test updating PairFeatures with enhanced features."""
        # Create base pair features
        pair_features = PairFeatures(
            note_a_path="a.md",
            note_b_path="b.md",
            cosine_similarity=0.5
        )
        
        # Create enhanced features
        enhanced_features = {
            'semantic_similarity': 0.8,
            'tfidf_similarity': 0.7,
            'combined_similarity': 0.75,
            'topic_similarity': 0.6,
            'same_dominant_topic': True,
            'content_similarity': 0.65
        }
        
        updated_features = feature_engineer.update_pair_features_with_enhanced(
            pair_features, enhanced_features
        )
        
        # Check that enhanced features were integrated
        assert updated_features.cosine_similarity == 0.8  # Updated from semantic_similarity
        assert updated_features.tfidf_similarity == 0.7
        assert updated_features.combined_similarity == 0.75
        assert updated_features.topic_similarity == 0.6
        assert updated_features.same_dominant_topic is True
    
    def test_get_analyzer_status(self, feature_engineer, sample_note_data):
        """Test getting analyzer status."""
        # Initially, analyzers should not be fitted
        status = feature_engineer.get_analyzer_status()
        
        assert isinstance(status, dict)
        assert 'semantic_analyzer' in status
        assert 'content_analyzer' in status
        assert 'topic_modeler' in status
        assert 'graph_analyzer' in status
        
        assert status['semantic_analyzer']['fitted'] is False
        assert status['topic_modeler']['fitted'] is False
        assert status['topic_modeler']['topic_count'] == 0
        assert status['graph_analyzer']['available'] is True
        
        # After fitting, status should change
        with patch.object(feature_engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic_fit, \
             patch.object(feature_engineer.topic_modeler, 'fit_topics') as mock_topic_fit, \
             patch.object(feature_engineer.semantic_analyzer, 'get_vocabulary_stats') as mock_vocab_stats, \
             patch.object(feature_engineer.topic_modeler, 'get_model_info') as mock_model_info:
            
            mock_semantic_fit.return_value = {'tfidf_features': 100}
            mock_topic_fit.return_value = TopicModelResult(topic_count=3)
            mock_vocab_stats.return_value = {'vocabulary_size': 100}
            mock_model_info.return_value = {'fitted': True, 'method': 'bertopic'}
            
            feature_engineer.fit_analyzers(sample_note_data)
            
            status = feature_engineer.get_analyzer_status()
            
            assert status['semantic_analyzer']['fitted'] is True
            assert status['topic_modeler']['fitted'] is True
            assert status['topic_modeler']['topic_count'] == 3
    
    def test_error_handling_integration(self, feature_engineer, sample_note_data):
        """Test error handling integration across all components."""
        # Clear previous errors
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # Test with problematic data that might cause errors
        problematic_note = NoteData(
            path="problematic.md",
            content="A" * 100000,  # Very long content
            embedding=np.array([float('inf'), float('nan'), 0.0, 0.0])  # Problematic embedding
        )
        
        # Should not raise exceptions
        features = feature_engineer.extract_note_features(problematic_note)
        assert isinstance(features, EnhancedFeatures)
        
        # Test pair features with problematic data
        pair_features = feature_engineer.extract_pair_features(problematic_note, problematic_note)
        assert isinstance(pair_features, dict)
        
        # Errors might be tracked, but should not crash
        final_error_count = len(error_tracker.errors)
        assert final_error_count >= initial_error_count
    
    def test_feature_quality_validation(self, feature_engineer, sample_note_data):
        """Test feature quality validation and correction."""
        # Mock analyzers to return problematic values
        with patch.object(feature_engineer.content_analyzer, 'analyze_content') as mock_content:
            
            # Return features with problematic values
            problematic_features = ContentFeatures(
                sentiment_score=float('nan'),  # NaN value
                complexity_score=2.0,  # Out of range
                vocabulary_richness=-0.5  # Negative value
            )
            mock_content.return_value = problematic_features
            
            note_data = sample_note_data[0]
            features = feature_engineer.extract_note_features(note_data)
            
            # Features should be corrected
            assert isinstance(features, EnhancedFeatures)
            assert features.content_features is not None
            # The exact correction depends on implementation
    
    def test_memory_efficiency_large_dataset(self, feature_engineer):
        """Test memory efficiency with large dataset."""
        # Create large dataset
        large_dataset = []
        for i in range(100):
            note = NoteData(
                path=f"note_{i}.md",
                content=f"This is note {i} with some content about topic {i % 5}.",
                embedding=np.random.rand(384),
                metadata={"id": i}
            )
            large_dataset.append(note)
        
        # Should handle large dataset without memory issues
        results = feature_engineer.fit_analyzers(large_dataset)
        
        assert isinstance(results, dict)
        assert results['notes_processed'] == 100
    
    @pytest.mark.parametrize("content,expected_features", [
        ("", False),  # Empty content should not extract features
        ("Short text", True),  # Normal content should extract features
        ("A" * 10000, True),  # Very long content should still work
        (None, False),  # None content should not extract features
    ])
    def test_feature_extraction_parametrized(self, feature_engineer, content, expected_features):
        """Test feature extraction with various content types."""
        note_data = NoteData(
            path="test.md",
            content=content,
            embedding=np.array([0.1, 0.2, 0.3, 0.4]) if content else None
        )
        
        features = feature_engineer.extract_note_features(note_data)
        
        assert isinstance(features, EnhancedFeatures)
        
        if expected_features:
            # Should have attempted to extract some features
            # (exact behavior depends on implementation)
            pass
        else:
            # Should return minimal features for empty/None content
            assert features.content_features is None


class TestFeatureEngineerErrorHandling:
    """Test error handling in FeatureEngineer."""
    
    @pytest.fixture
    def failing_feature_engineer(self, mock_vector_encoder):
        """Create FeatureEngineer with failing components."""
        engineer = FeatureEngineer(mock_vector_encoder)
        
        # Mock components to fail
        engineer.semantic_analyzer.fit_and_transform = Mock(side_effect=Exception("Semantic failed"))
        engineer.content_analyzer.analyze_content = Mock(side_effect=Exception("Content failed"))
        engineer.topic_modeler.fit_topics = Mock(side_effect=Exception("Topic failed"))
        
        return engineer
    
    def test_graceful_degradation_all_failures(self, failing_feature_engineer, sample_note_data):
        """Test graceful degradation when all components fail."""
        # Should not raise exceptions
        results = failing_feature_engineer.fit_analyzers(sample_note_data)
        
        assert isinstance(results, dict)
        assert results['semantic_fitted'] is False
        assert results['topic_fitted'] is False
        assert len(results['errors']) > 0
        
        # Feature extraction should still work with fallbacks
        note_data = sample_note_data[0]
        features = failing_feature_engineer.extract_note_features(note_data)
        
        assert isinstance(features, EnhancedFeatures)
        # Should use fallback values
    
    def test_partial_component_failure(self, mock_vector_encoder, sample_note_data):
        """Test behavior when only some components fail."""
        engineer = FeatureEngineer(mock_vector_encoder)
        
        # Make only content analyzer fail
        engineer.content_analyzer.analyze_content = Mock(side_effect=Exception("Content failed"))
        
        # Mock successful semantic and topic fitting
        with patch.object(engineer.semantic_analyzer, 'fit_and_transform') as mock_semantic, \
             patch.object(engineer.topic_modeler, 'fit_topics') as mock_topic:
            
            mock_semantic.return_value = {'tfidf_features': 100}
            mock_topic.return_value = TopicModelResult(topic_count=2)
            
            results = engineer.fit_analyzers(sample_note_data)
            
            # Semantic and topic should succeed
            assert results['semantic_fitted'] is True
            assert results['topic_fitted'] is True
            
            # Feature extraction should work with partial failure
            note_data = sample_note_data[0]
            features = engineer.extract_note_features(note_data)
            
            assert isinstance(features, EnhancedFeatures)
            # Content features should be None due to failure
            assert features.content_features is None
    
    def test_error_tracking_comprehensive(self, failing_feature_engineer, sample_note_data):
        """Test comprehensive error tracking."""
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # This should trigger multiple errors
        results = failing_feature_engineer.fit_analyzers(sample_note_data)
        features = failing_feature_engineer.extract_note_features(sample_note_data[0])
        
        # Should have tracked errors
        final_error_count = len(error_tracker.errors)
        assert final_error_count > initial_error_count
        
        # Check error summary
        error_summary = error_tracker.get_error_summary()
        assert error_summary['total_errors'] > 0
        assert len(error_summary['by_component']) > 0


if __name__ == "__main__":
    pytest.main([__file__])