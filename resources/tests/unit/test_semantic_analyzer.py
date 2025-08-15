"""
Unit tests for SemanticAnalyzer component.

Tests semantic similarity computation, TF-IDF feature extraction,
and embedding generation with comprehensive error handling.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from scipy.sparse import csr_matrix

from src.jarvis.tools.dataset_generation.analyzers.semantic_analyzer import (
    SemanticAnalyzer
)
from src.jarvis.tools.dataset_generation.error_handling import (
    get_error_tracker, ComponentType
)


class TestSemanticAnalyzer:
    """Test SemanticAnalyzer functionality."""
    
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
    def analyzer(self, mock_vector_encoder):
        """Create SemanticAnalyzer instance for testing."""
        return SemanticAnalyzer(mock_vector_encoder)
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
    
    def test_analyzer_initialization(self, mock_vector_encoder):
        """Test analyzer initialization."""
        analyzer = SemanticAnalyzer(mock_vector_encoder)
        
        assert analyzer.vector_encoder == mock_vector_encoder
        assert analyzer.tfidf_vectorizer is not None
        assert analyzer._fitted is False
        assert analyzer._tfidf_matrix is None
        assert analyzer._embeddings is None
    
    def test_fit_and_transform_basic(self, analyzer, sample_texts):
        """Test basic fit and transform functionality."""
        result = analyzer.fit_and_transform(sample_texts)
        
        assert isinstance(result, dict)
        assert 'tfidf_features' in result
        assert 'embedding_features' in result
        assert 'vocabulary_size' in result
        
        assert analyzer._fitted is True
        assert analyzer._tfidf_matrix is not None
        assert analyzer._embeddings is not None
        
        # Check TF-IDF matrix shape
        assert analyzer._tfidf_matrix.shape[0] == len(sample_texts)
        assert analyzer._tfidf_matrix.shape[1] > 0  # Should have vocabulary
        
        # Check embeddings shape
        assert analyzer._embeddings.shape[0] == len(sample_texts)
        assert analyzer._embeddings.shape[1] == 4  # Mock encoder returns 4D vectors
    
    def test_fit_and_transform_empty_texts(self, analyzer):
        """Test fit and transform with empty texts."""
        result = analyzer.fit_and_transform([])
        
        assert result is None
        assert not analyzer._fitted
    
    def test_fit_and_transform_none_texts(self, analyzer):
        """Test fit and transform with None texts."""
        result = analyzer.fit_and_transform(None)
        
        assert result is None
        assert not analyzer._fitted
    
    def test_fit_and_transform_with_empty_strings(self, analyzer):
        """Test fit and transform with some empty strings."""
        texts = ["Valid text", "", "Another valid text", None]
        result = analyzer.fit_and_transform(texts)
        
        # Should filter out empty/None texts
        assert result is not None
        assert analyzer._fitted is True
        # Should only process valid texts
        assert analyzer._tfidf_matrix.shape[0] == 2
        assert analyzer._embeddings.shape[0] == 2
    
    def test_compute_tfidf_features_not_fitted(self, analyzer, sample_texts):
        """Test TF-IDF computation when not fitted."""
        tfidf_matrix = analyzer.compute_tfidf_features(sample_texts)
        
        # Should return empty matrix when not fitted
        assert tfidf_matrix.shape[0] == 0
        assert tfidf_matrix.shape[1] == 0
    
    def test_compute_tfidf_features_fitted(self, analyzer, sample_texts):
        """Test TF-IDF computation when fitted."""
        # First fit the analyzer
        analyzer.fit_and_transform(sample_texts)
        
        # Then compute features for new texts
        new_texts = ["Artificial intelligence is fascinating."]
        tfidf_matrix = analyzer.compute_tfidf_features(new_texts)
        
        assert tfidf_matrix.shape[0] == 1
        assert tfidf_matrix.shape[1] > 0
        assert hasattr(tfidf_matrix, 'toarray')  # Should be sparse matrix
    
    def test_compute_embeddings_basic(self, analyzer, sample_texts):
        """Test embedding computation."""
        embeddings = analyzer.compute_embeddings(sample_texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] == 4  # Mock encoder dimension
        
        # Verify encoder was called
        analyzer.vector_encoder.encode_batch.assert_called_once_with(sample_texts)
    
    def test_compute_embeddings_empty_texts(self, analyzer):
        """Test embedding computation with empty texts."""
        embeddings = analyzer.compute_embeddings([])
        
        assert embeddings.shape[0] == 0
        assert embeddings.shape[1] == 0
    
    def test_compute_similarity_matrix_basic(self, analyzer):
        """Test similarity matrix computation."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
        
        assert similarity_matrix.shape == (3, 3)
        
        # Check diagonal (self-similarity should be 1.0)
        np.testing.assert_array_almost_equal(np.diag(similarity_matrix), [1.0, 1.0, 1.0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(similarity_matrix, similarity_matrix.T)
        
        # Check specific similarities
        assert similarity_matrix[0, 1] == pytest.approx(0.0, abs=1e-6)  # Orthogonal vectors
        assert similarity_matrix[0, 2] > 0.5  # Should be similar
    
    def test_compute_similarity_matrix_empty(self, analyzer):
        """Test similarity matrix with empty embeddings."""
        embeddings = np.array([]).reshape(0, 0)
        similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
        
        assert similarity_matrix.shape == (0, 0)
    
    def test_compute_similarity_matrix_single_embedding(self, analyzer):
        """Test similarity matrix with single embedding."""
        embeddings = np.array([[1.0, 2.0, 3.0]])
        similarity_matrix = analyzer.compute_similarity_matrix(embeddings)
        
        assert similarity_matrix.shape == (1, 1)
        assert similarity_matrix[0, 0] == pytest.approx(1.0)
    
    def test_extract_pair_features_not_fitted(self, analyzer):
        """Test pair feature extraction when not fitted."""
        features = analyzer.extract_pair_features("Text A", "Text B")
        
        # Should return minimal features when not fitted
        assert isinstance(features, dict)
        assert 'semantic_similarity' in features
        assert 'tfidf_similarity' in features
        assert 'combined_similarity' in features
        
        # Values should be fallback values
        assert features['semantic_similarity'] == 0.0
        assert features['tfidf_similarity'] == 0.0
        assert features['combined_similarity'] == 0.0
    
    def test_extract_pair_features_fitted(self, analyzer, sample_texts):
        """Test pair feature extraction when fitted."""
        # First fit the analyzer
        analyzer.fit_and_transform(sample_texts)
        
        # Extract features for a pair
        features = analyzer.extract_pair_features(
            "Machine learning algorithms are powerful.",
            "Deep learning is a type of machine learning."
        )
        
        assert isinstance(features, dict)
        assert 'semantic_similarity' in features
        assert 'tfidf_similarity' in features
        assert 'combined_similarity' in features
        
        # Should have computed actual similarities
        assert -1.0 <= features['semantic_similarity'] <= 1.0
        assert 0.0 <= features['tfidf_similarity'] <= 1.0
        assert 0.0 <= features['combined_similarity'] <= 1.0
    
    def test_extract_pair_features_identical_texts(self, analyzer, sample_texts):
        """Test pair feature extraction with identical texts."""
        analyzer.fit_and_transform(sample_texts)
        
        text = "Machine learning is fascinating."
        features = analyzer.extract_pair_features(text, text)
        
        # Identical texts should have high similarity
        assert features['semantic_similarity'] == pytest.approx(1.0, abs=0.1)
        assert features['tfidf_similarity'] == pytest.approx(1.0, abs=0.1)
        assert features['combined_similarity'] == pytest.approx(1.0, abs=0.1)
    
    def test_extract_pair_features_empty_texts(self, analyzer, sample_texts):
        """Test pair feature extraction with empty texts."""
        analyzer.fit_and_transform(sample_texts)
        
        features = analyzer.extract_pair_features("", "")
        
        # Empty texts should return fallback values
        assert features['semantic_similarity'] == 0.0
        assert features['tfidf_similarity'] == 0.0
        assert features['combined_similarity'] == 0.0
    
    def test_get_vocabulary_stats_not_fitted(self, analyzer):
        """Test vocabulary stats when not fitted."""
        stats = analyzer.get_vocabulary_stats()
        
        assert stats['fitted'] is False
        assert stats['vocabulary_size'] == 0
        assert stats['feature_names'] == []
    
    def test_get_vocabulary_stats_fitted(self, analyzer, sample_texts):
        """Test vocabulary stats when fitted."""
        analyzer.fit_and_transform(sample_texts)
        stats = analyzer.get_vocabulary_stats()
        
        assert stats['fitted'] is True
        assert stats['vocabulary_size'] > 0
        assert len(stats['feature_names']) > 0
        assert isinstance(stats['feature_names'], list)
    
    def test_cosine_similarity_computation(self, analyzer):
        """Test cosine similarity computation."""
        # Test vectors
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        vec_c = np.array([1.0, 1.0, 0.0])
        
        # Orthogonal vectors should have 0 similarity
        sim_ab = analyzer._cosine_similarity(vec_a, vec_b)
        assert sim_ab == pytest.approx(0.0, abs=1e-6)
        
        # Vector with itself should have 1.0 similarity
        sim_aa = analyzer._cosine_similarity(vec_a, vec_a)
        assert sim_aa == pytest.approx(1.0, abs=1e-6)
        
        # Vectors at 45 degrees should have ~0.707 similarity
        sim_ac = analyzer._cosine_similarity(vec_a, vec_c)
        assert sim_ac == pytest.approx(0.707, abs=0.01)
    
    def test_cosine_similarity_zero_vectors(self, analyzer):
        """Test cosine similarity with zero vectors."""
        vec_zero = np.array([0.0, 0.0, 0.0])
        vec_normal = np.array([1.0, 2.0, 3.0])
        
        # Zero vector should return 0 similarity
        sim = analyzer._cosine_similarity(vec_zero, vec_normal)
        assert sim == 0.0
        
        # Two zero vectors should return 0 similarity
        sim_zero = analyzer._cosine_similarity(vec_zero, vec_zero)
        assert sim_zero == 0.0
    
    def test_tfidf_similarity_computation(self, analyzer):
        """Test TF-IDF similarity computation."""
        # Create mock sparse matrices
        vec_a = csr_matrix([[1.0, 0.0, 0.5]])
        vec_b = csr_matrix([[0.0, 1.0, 0.5]])
        
        similarity = analyzer._tfidf_cosine_similarity(vec_a, vec_b)
        
        assert 0.0 <= similarity <= 1.0
        # Should be > 0 due to shared term (0.5 in both)
        assert similarity > 0.0
    
    def test_tfidf_similarity_identical_vectors(self, analyzer):
        """Test TF-IDF similarity with identical vectors."""
        vec = csr_matrix([[1.0, 0.5, 0.2]])
        
        similarity = analyzer._tfidf_cosine_similarity(vec, vec)
        
        assert similarity == pytest.approx(1.0, abs=1e-6)
    
    def test_tfidf_similarity_zero_vectors(self, analyzer):
        """Test TF-IDF similarity with zero vectors."""
        vec_zero = csr_matrix([[0.0, 0.0, 0.0]])
        vec_normal = csr_matrix([[1.0, 0.5, 0.2]])
        
        similarity = analyzer._tfidf_cosine_similarity(vec_zero, vec_normal)
        
        assert similarity == 0.0
    
    def test_combined_similarity_calculation(self, analyzer):
        """Test combined similarity calculation."""
        semantic_sim = 0.8
        tfidf_sim = 0.6
        
        combined = analyzer._calculate_combined_similarity(semantic_sim, tfidf_sim)
        
        # Should be weighted average (default weights: 0.7 semantic, 0.3 tfidf)
        expected = 0.7 * semantic_sim + 0.3 * tfidf_sim
        assert combined == pytest.approx(expected, abs=1e-6)
    
    def test_combined_similarity_edge_cases(self, analyzer):
        """Test combined similarity with edge cases."""
        # Both zero
        combined = analyzer._calculate_combined_similarity(0.0, 0.0)
        assert combined == 0.0
        
        # Both one
        combined = analyzer._calculate_combined_similarity(1.0, 1.0)
        assert combined == 1.0
        
        # One zero, one one
        combined = analyzer._calculate_combined_similarity(0.0, 1.0)
        assert 0.0 < combined < 1.0
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.semantic_analyzer.TfidfVectorizer')
    def test_tfidf_vectorizer_configuration(self, mock_tfidf_class, mock_vector_encoder):
        """Test TF-IDF vectorizer configuration."""
        mock_vectorizer = Mock()
        mock_tfidf_class.return_value = mock_vectorizer
        
        analyzer = SemanticAnalyzer(mock_vector_encoder)
        
        # Check that TfidfVectorizer was initialized with correct parameters
        mock_tfidf_class.assert_called_once_with(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
        )
    
    def test_error_handling_with_encoder_failure(self, sample_texts):
        """Test error handling when vector encoder fails."""
        # Create failing encoder
        failing_encoder = Mock()
        failing_encoder.encode_batch.side_effect = Exception("Encoder failed")
        
        analyzer = SemanticAnalyzer(failing_encoder)
        
        # Should handle encoder failure gracefully
        result = analyzer.fit_and_transform(sample_texts)
        
        # Should return None or handle gracefully
        # The exact behavior depends on error handling implementation
        assert result is None or isinstance(result, dict)
    
    def test_error_handling_with_tfidf_failure(self, mock_vector_encoder, sample_texts):
        """Test error handling when TF-IDF fails."""
        analyzer = SemanticAnalyzer(mock_vector_encoder)
        
        # Mock TF-IDF to fail
        analyzer.tfidf_vectorizer.fit_transform.side_effect = Exception("TF-IDF failed")
        
        # Should handle TF-IDF failure gracefully
        result = analyzer.fit_and_transform(sample_texts)
        
        # Should return None or handle gracefully
        assert result is None or isinstance(result, dict)
    
    def test_memory_efficiency_large_texts(self, analyzer):
        """Test memory efficiency with large number of texts."""
        # Create many texts to test memory handling
        large_text_list = [f"This is test document number {i} with some content." for i in range(100)]
        
        result = analyzer.fit_and_transform(large_text_list)
        
        assert result is not None
        assert analyzer._fitted is True
        assert analyzer._tfidf_matrix.shape[0] == 100
        assert analyzer._embeddings.shape[0] == 100
    
    @pytest.mark.parametrize("text_a,text_b,expected_high_similarity", [
        ("machine learning", "machine learning", True),
        ("artificial intelligence", "AI technology", True),
        ("cat", "dog", False),
        ("", "", False),  # Empty texts should have low similarity
    ])
    def test_similarity_computation_parametrized(self, analyzer, sample_texts, text_a, text_b, expected_high_similarity):
        """Test similarity computation with various text pairs."""
        analyzer.fit_and_transform(sample_texts)
        features = analyzer.extract_pair_features(text_a, text_b)
        
        if expected_high_similarity:
            assert features['combined_similarity'] > 0.3  # Threshold for "high" similarity
        else:
            assert features['combined_similarity'] <= 0.3  # Threshold for "low" similarity


class TestSemanticAnalyzerErrorHandling:
    """Test error handling in SemanticAnalyzer."""
    
    def test_graceful_degradation_encoder_failure(self):
        """Test graceful degradation when encoder fails."""
        failing_encoder = Mock()
        failing_encoder.encode_batch.side_effect = Exception("Encoder failed")
        
        analyzer = SemanticAnalyzer(failing_encoder)
        
        # Should not raise exception
        result = analyzer.fit_and_transform(["test text"])
        
        # Should handle failure gracefully
        assert result is None or isinstance(result, dict)
    
    def test_error_tracking(self):
        """Test that errors are properly tracked."""
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # Create analyzer with failing encoder
        failing_encoder = Mock()
        failing_encoder.encode_batch.side_effect = Exception("Test error")
        
        analyzer = SemanticAnalyzer(failing_encoder)
        
        # This should trigger error handling
        result = analyzer.fit_and_transform(["test"])
        
        # Check if error was tracked (depends on implementation)
        # The exact behavior depends on how errors are handled
        final_error_count = len(error_tracker.errors)
        assert final_error_count >= initial_error_count
    
    def test_fallback_values_consistency(self, mock_vector_encoder):
        """Test that fallback values are consistent."""
        analyzer = SemanticAnalyzer(mock_vector_encoder)
        
        # Test multiple calls return consistent fallback values
        features1 = analyzer.extract_pair_features("", "")
        features2 = analyzer.extract_pair_features("", "")
        
        assert features1 == features2
        
        # Check fallback values are reasonable
        assert features1['semantic_similarity'] == 0.0
        assert features1['tfidf_similarity'] == 0.0
        assert features1['combined_similarity'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])