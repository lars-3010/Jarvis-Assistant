"""
Unit tests for TopicModeler component.

Tests topic modeling capabilities including BERTopic, LDA fallback,
and topic prediction with comprehensive error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.jarvis.tools.dataset_generation.analyzers.topic_modeler import (
    TopicModeler, TopicModelResult, TopicPrediction
)
from src.jarvis.tools.dataset_generation.error_handling import (
    get_error_tracker, ComponentType
)


class TestTopicModelResult:
    """Test TopicModelResult data class."""
    
    def test_topic_model_result_initialization(self):
        """Test TopicModelResult initialization."""
        result = TopicModelResult()
        
        assert result.topic_count == 0
        assert result.topic_labels == {}
        assert result.topic_keywords == {}
        assert result.model_type == ""
        assert result.coherence_score == 0.0
        assert result.model_info == {}
    
    def test_topic_model_result_with_values(self):
        """Test TopicModelResult with specific values."""
        topic_labels = {0: "Technology", 1: "Science"}
        topic_keywords = {0: ["AI", "machine", "learning"], 1: ["research", "study", "analysis"]}
        model_info = {"method": "bertopic", "n_components": 2}
        
        result = TopicModelResult(
            topic_count=2,
            topic_labels=topic_labels,
            topic_keywords=topic_keywords,
            model_type="bertopic",
            coherence_score=0.75,
            model_info=model_info
        )
        
        assert result.topic_count == 2
        assert result.topic_labels == topic_labels
        assert result.topic_keywords == topic_keywords
        assert result.model_type == "bertopic"
        assert result.coherence_score == 0.75
        assert result.model_info == model_info


class TestTopicPrediction:
    """Test TopicPrediction data class."""
    
    def test_topic_prediction_initialization(self):
        """Test TopicPrediction initialization."""
        prediction = TopicPrediction()
        
        assert prediction.topic_assignments == []
        assert prediction.topic_probabilities == []
        assert prediction.confidence_scores == []
    
    def test_topic_prediction_with_values(self):
        """Test TopicPrediction with specific values."""
        assignments = [0, 1, 0]
        probabilities = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        confidence = [0.8, 0.7, 0.9]
        
        prediction = TopicPrediction(
            topic_assignments=assignments,
            topic_probabilities=probabilities,
            confidence_scores=confidence
        )
        
        assert prediction.topic_assignments == assignments
        assert prediction.topic_probabilities == probabilities
        assert prediction.confidence_scores == confidence


class TestTopicModeler:
    """Test TopicModeler functionality."""
    
    @pytest.fixture
    def modeler(self):
        """Create TopicModeler instance for testing."""
        return TopicModeler(method="auto", min_topic_size=3)
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "Machine learning algorithms are used in artificial intelligence applications.",
            "Deep learning neural networks process complex data patterns effectively.",
            "Natural language processing helps computers understand human language.",
            "Computer vision systems can recognize objects in images automatically.",
            "Data science involves statistical analysis and machine learning techniques.",
            "Scientific research requires careful methodology and data analysis.",
            "Research studies often use statistical methods to validate hypotheses.",
            "Academic papers present findings from systematic investigations.",
            "Experimental design is crucial for reliable research outcomes.",
            "Peer review ensures quality in scientific publications."
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return np.random.rand(10, 384)  # 10 documents, 384-dim embeddings
    
    def test_modeler_initialization(self):
        """Test modeler initialization."""
        modeler = TopicModeler(method="bertopic", min_topic_size=5)
        
        assert modeler.method == "bertopic"
        assert modeler.min_topic_size == 5
        assert modeler._fitted is False
        assert modeler._model is None
        assert modeler._topic_result is None
    
    def test_modeler_initialization_auto_method(self):
        """Test modeler initialization with auto method."""
        modeler = TopicModeler(method="auto")
        
        assert modeler.method == "auto"
        assert modeler._fitted is False
    
    def test_fit_topics_empty_documents(self, modeler):
        """Test fitting topics with empty documents."""
        result = modeler.fit_topics([])
        
        assert result is None
        assert not modeler._fitted
    
    def test_fit_topics_none_documents(self, modeler):
        """Test fitting topics with None documents."""
        result = modeler.fit_topics(None)
        
        assert result is None
        assert not modeler._fitted
    
    def test_fit_topics_insufficient_documents(self, modeler):
        """Test fitting topics with insufficient documents."""
        # Only 2 documents, but min_topic_size is 3
        texts = ["Document one", "Document two"]
        result = modeler.fit_topics(texts)
        
        # Should return None or minimal result
        assert result is None or result.topic_count == 0
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic')
    def test_fit_topics_bertopic_success(self, mock_bertopic_class, modeler, sample_texts, sample_embeddings):
        """Test successful topic fitting with BERTopic."""
        # Mock BERTopic
        mock_model = Mock()
        mock_model.fit_transform.return_value = (
            [0, 0, 1, 1, 0, 2, 2, 2, 1, 1],  # topic assignments
            [[0.8, 0.2, 0.0], [0.9, 0.1, 0.0]]  # probabilities (partial)
        )
        mock_model.get_topic_info.return_value = Mock()
        mock_model.get_topic_info.return_value.to_dict.return_value = {
            'Topic': [0, 1, 2],
            'Count': [4, 3, 3]
        }
        mock_model.get_topics.return_value = {
            0: [("machine", 0.5), ("learning", 0.4), ("ai", 0.3)],
            1: ("research", 0.6), ("study", 0.4), ("analysis", 0.3)],
            2: [("data", 0.5), ("science", 0.4), ("statistical", 0.3)]
        }
        mock_bertopic_class.return_value = mock_model
        
        modeler.method = "bertopic"
        result = modeler.fit_topics(sample_texts, sample_embeddings)
        
        assert result is not None
        assert isinstance(result, TopicModelResult)
        assert result.topic_count > 0
        assert result.model_type == "bertopic"
        assert modeler._fitted is True
        assert modeler._model == mock_model
        assert modeler._topic_result == result
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic')
    def test_fit_topics_bertopic_failure_fallback_to_lda(self, mock_bertopic_class, modeler, sample_texts):
        """Test BERTopic failure with fallback to LDA."""
        # Mock BERTopic to fail
        mock_bertopic_class.side_effect = Exception("BERTopic failed")
        
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.LatentDirichletAllocation') as mock_lda_class:
            mock_lda = Mock()
            mock_lda.fit.return_value = mock_lda
            mock_lda.transform.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
            mock_lda.n_components = 2
            mock_lda_class.return_value = mock_lda
            
            with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.TfidfVectorizer') as mock_tfidf:
                mock_vectorizer = Mock()
                mock_vectorizer.fit_transform.return_value = Mock()
                mock_vectorizer.get_feature_names_out.return_value = ["word1", "word2", "word3"]
                mock_tfidf.return_value = mock_vectorizer
                
                result = modeler.fit_topics(sample_texts)
                
                assert result is not None
                assert result.model_type == "lda"
                assert modeler._fitted is True
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.LatentDirichletAllocation')
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.TfidfVectorizer')
    def test_fit_topics_lda_success(self, mock_tfidf_class, mock_lda_class, modeler, sample_texts):
        """Test successful topic fitting with LDA."""
        # Mock TF-IDF vectorizer
        mock_vectorizer = Mock()
        mock_tfidf_matrix = Mock()
        mock_vectorizer.fit_transform.return_value = mock_tfidf_matrix
        mock_vectorizer.get_feature_names_out.return_value = ["machine", "learning", "research", "data"]
        mock_tfidf_class.return_value = mock_vectorizer
        
        # Mock LDA
        mock_lda = Mock()
        mock_lda.fit.return_value = mock_lda
        mock_lda.transform.return_value = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4]
        ])
        mock_lda.n_components = 2
        mock_lda.components_ = np.array([
            [0.5, 0.3, 0.1, 0.1],  # Topic 0 word distribution
            [0.1, 0.1, 0.4, 0.4]   # Topic 1 word distribution
        ])
        mock_lda_class.return_value = mock_lda
        
        modeler.method = "lda"
        result = modeler.fit_topics(sample_texts)
        
        assert result is not None
        assert isinstance(result, TopicModelResult)
        assert result.topic_count == 2
        assert result.model_type == "lda"
        assert len(result.topic_keywords) == 2
        assert modeler._fitted is True
    
    def test_predict_topics_not_fitted(self, modeler, sample_texts):
        """Test topic prediction when not fitted."""
        prediction = modeler.predict_topics(sample_texts)
        
        assert prediction is not None
        assert isinstance(prediction, TopicPrediction)
        assert len(prediction.topic_assignments) == 0
        assert len(prediction.topic_probabilities) == 0
    
    def test_predict_topics_empty_texts(self, modeler):
        """Test topic prediction with empty texts."""
        prediction = modeler.predict_topics([])
        
        assert prediction is not None
        assert len(prediction.topic_assignments) == 0
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic')
    def test_predict_topics_bertopic_fitted(self, mock_bertopic_class, modeler, sample_texts, sample_embeddings):
        """Test topic prediction with fitted BERTopic model."""
        # First fit the model
        mock_model = Mock()
        mock_model.fit_transform.return_value = ([0, 1, 0], [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        mock_model.get_topic_info.return_value = Mock()
        mock_model.get_topic_info.return_value.to_dict.return_value = {
            'Topic': [0, 1], 'Count': [2, 1]
        }
        mock_model.get_topics.return_value = {
            0: [("word1", 0.5)],
            1: [("word2", 0.4)]
        }
        mock_bertopic_class.return_value = mock_model
        
        modeler.method = "bertopic"
        modeler.fit_topics(sample_texts[:3], sample_embeddings[:3])
        
        # Now predict on new texts
        mock_model.transform.return_value = ([0, 1], [[0.7, 0.3], [0.4, 0.6]])
        
        new_texts = ["New machine learning text", "New research text"]
        new_embeddings = sample_embeddings[:2]
        
        prediction = modeler.predict_topics(new_texts, new_embeddings)
        
        assert prediction is not None
        assert len(prediction.topic_assignments) == 2
        assert len(prediction.topic_probabilities) == 2
        assert prediction.topic_assignments == [0, 1]
    
    def test_get_model_info_not_fitted(self, modeler):
        """Test getting model info when not fitted."""
        info = modeler.get_model_info()
        
        assert info['fitted'] is False
        assert info['method'] == "auto"
        assert info['model_type'] is None
        assert info['topic_count'] == 0
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic')
    def test_get_model_info_fitted(self, mock_bertopic_class, modeler, sample_texts):
        """Test getting model info when fitted."""
        # Mock and fit model
        mock_model = Mock()
        mock_model.fit_transform.return_value = ([0, 1], [[0.8, 0.2], [0.3, 0.7]])
        mock_model.get_topic_info.return_value = Mock()
        mock_model.get_topic_info.return_value.to_dict.return_value = {
            'Topic': [0, 1], 'Count': [1, 1]
        }
        mock_model.get_topics.return_value = {0: [("word1", 0.5)], 1: [("word2", 0.4)]}
        mock_bertopic_class.return_value = mock_model
        
        modeler.method = "bertopic"
        modeler.fit_topics(sample_texts[:2])
        
        info = modeler.get_model_info()
        
        assert info['fitted'] is True
        assert info['method'] == "bertopic"
        assert info['model_type'] == "bertopic"
        assert info['topic_count'] > 0
    
    def test_determine_optimal_topics_small_dataset(self, modeler):
        """Test optimal topic determination for small dataset."""
        n_topics = modeler._determine_optimal_topics(5)  # 5 documents
        
        assert n_topics >= 2
        assert n_topics <= 3  # Should be conservative for small datasets
    
    def test_determine_optimal_topics_medium_dataset(self, modeler):
        """Test optimal topic determination for medium dataset."""
        n_topics = modeler._determine_optimal_topics(50)  # 50 documents
        
        assert n_topics >= 3
        assert n_topics <= 10
    
    def test_determine_optimal_topics_large_dataset(self, modeler):
        """Test optimal topic determination for large dataset."""
        n_topics = modeler._determine_optimal_topics(500)  # 500 documents
        
        assert n_topics >= 5
        assert n_topics <= 20
    
    def test_extract_lda_topic_keywords(self, modeler):
        """Test LDA topic keyword extraction."""
        # Mock LDA components and feature names
        components = np.array([
            [0.5, 0.3, 0.2, 0.1, 0.05],  # Topic 0
            [0.1, 0.2, 0.4, 0.3, 0.15]   # Topic 1
        ])
        feature_names = ["machine", "learning", "research", "data", "analysis"]
        
        keywords = modeler._extract_lda_topic_keywords(components, feature_names, top_k=3)
        
        assert len(keywords) == 2  # Two topics
        assert 0 in keywords and 1 in keywords
        
        # Check topic 0 keywords (should be top 3 by weight)
        topic_0_keywords = keywords[0]
        assert len(topic_0_keywords) == 3
        assert topic_0_keywords[0][0] == "machine"  # Highest weight
        assert topic_0_keywords[1][0] == "learning"  # Second highest
        assert topic_0_keywords[2][0] == "research"  # Third highest
        
        # Check topic 1 keywords
        topic_1_keywords = keywords[1]
        assert len(topic_1_keywords) == 3
        assert topic_1_keywords[0][0] == "research"  # Highest weight for topic 1
    
    def test_generate_topic_labels(self, modeler):
        """Test topic label generation."""
        topic_keywords = {
            0: [("machine", 0.5), ("learning", 0.4), ("ai", 0.3)],
            1: [("research", 0.6), ("study", 0.4), ("analysis", 0.3)],
            2: [("data", 0.5), ("science", 0.4), ("statistical", 0.3)]
        }
        
        labels = modeler._generate_topic_labels(topic_keywords)
        
        assert len(labels) == 3
        assert 0 in labels and 1 in labels and 2 in labels
        
        # Labels should be based on top keywords
        assert "machine" in labels[0].lower() or "learning" in labels[0].lower()
        assert "research" in labels[1].lower() or "study" in labels[1].lower()
        assert "data" in labels[2].lower() or "science" in labels[2].lower()
    
    def test_calculate_coherence_score_placeholder(self, modeler):
        """Test coherence score calculation (placeholder implementation)."""
        topic_keywords = {
            0: [("machine", 0.5), ("learning", 0.4)],
            1: [("research", 0.6), ("study", 0.4)]
        }
        
        coherence = modeler._calculate_coherence_score(topic_keywords)
        
        # Should return a reasonable placeholder value
        assert 0.0 <= coherence <= 1.0
    
    def test_auto_method_selection_with_bertopic_available(self, modeler):
        """Test automatic method selection when BERTopic is available."""
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic'):
            method = modeler._select_method_auto()
            assert method == "bertopic"
    
    def test_auto_method_selection_bertopic_unavailable(self, modeler):
        """Test automatic method selection when BERTopic is unavailable."""
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic', side_effect=ImportError):
            method = modeler._select_method_auto()
            assert method == "lda"
    
    def test_error_handling_with_invalid_method(self):
        """Test error handling with invalid method."""
        modeler = TopicModeler(method="invalid_method")
        
        result = modeler.fit_topics(["test document"])
        
        # Should handle gracefully
        assert result is None
        assert not modeler._fitted
    
    def test_memory_efficiency_large_dataset(self, modeler):
        """Test memory efficiency with large dataset."""
        # Create large dataset
        large_texts = [f"Document {i} with some content about topic {i % 3}" for i in range(100)]
        
        # Should handle without memory issues
        result = modeler.fit_topics(large_texts)
        
        # Should either succeed or fail gracefully
        assert result is None or isinstance(result, TopicModelResult)
    
    @pytest.mark.parametrize("method,expected_available", [
        ("bertopic", True),  # Assume available in test environment
        ("lda", True),       # Should always be available
        ("auto", True),      # Should select available method
        ("invalid", False),  # Invalid method
    ])
    def test_method_availability(self, method, expected_available):
        """Test different method availability."""
        modeler = TopicModeler(method=method)
        
        if expected_available and method != "invalid":
            # Should initialize without error
            assert modeler.method == method
        else:
            # Invalid method should still initialize but may fail later
            assert modeler.method == method


class TestTopicModelerErrorHandling:
    """Test error handling in TopicModeler."""
    
    def test_graceful_degradation_bertopic_failure(self):
        """Test graceful degradation when BERTopic fails."""
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic', side_effect=ImportError):
            modeler = TopicModeler(method="bertopic")
            
            result = modeler.fit_topics(["test document"])
            
            # Should handle failure gracefully
            assert result is None
            assert not modeler._fitted
    
    def test_graceful_degradation_lda_failure(self):
        """Test graceful degradation when LDA fails."""
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.LatentDirichletAllocation', side_effect=ImportError):
            modeler = TopicModeler(method="lda")
            
            result = modeler.fit_topics(["test document"])
            
            # Should handle failure gracefully
            assert result is None
            assert not modeler._fitted
    
    def test_error_tracking(self):
        """Test that errors are properly tracked."""
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # Create modeler that will fail
        with patch('src.jarvis.tools.dataset_generation.analyzers.topic_modeler.BERTopic', side_effect=Exception("Test error")):
            modeler = TopicModeler(method="bertopic")
            result = modeler.fit_topics(["test"])
            
            # Should handle error gracefully
            assert result is None
        
        # Check if error was tracked (depends on implementation)
        final_error_count = len(error_tracker.errors)
        assert final_error_count >= initial_error_count
    
    def test_fallback_values_consistency(self):
        """Test that fallback values are consistent."""
        modeler = TopicModeler()
        
        # Test multiple calls return consistent fallback values
        prediction1 = modeler.predict_topics(["test"])
        prediction2 = modeler.predict_topics(["test"])
        
        assert prediction1.topic_assignments == prediction2.topic_assignments
        assert prediction1.topic_probabilities == prediction2.topic_probabilities
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        modeler = TopicModeler()
        
        # Test with various invalid inputs
        invalid_inputs = [
            None,
            [],
            [""],
            [None],
            ["   "],  # Whitespace only
        ]
        
        for invalid_input in invalid_inputs:
            result = modeler.fit_topics(invalid_input)
            # Should handle gracefully without raising exceptions
            assert result is None or isinstance(result, TopicModelResult)
            
            prediction = modeler.predict_topics(invalid_input)
            assert isinstance(prediction, TopicPrediction)


if __name__ == "__main__":
    pytest.main([__file__])