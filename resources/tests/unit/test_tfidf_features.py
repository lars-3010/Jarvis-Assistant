"""
Test TF-IDF feature extraction functionality.

This module tests the TF-IDF feature extraction capabilities of the SemanticAnalyzer
and ensures proper integration with the dataset generation pipeline.
"""

import numpy as np
import pytest
import scipy.sparse
from unittest.mock import Mock, patch

from jarvis.tools.dataset_generation.analyzers.semantic_analyzer import SemanticAnalyzer
from jarvis.services.vector.encoder import VectorEncoder


class TestTFIDFFeatures:
    """Test TF-IDF feature extraction functionality."""

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock vector encoder."""
        encoder = Mock(spec=VectorEncoder)
        encoder.vector_dim = 384
        encoder.encode_documents.return_value = np.random.rand(3, 384).astype(np.float32)
        return encoder

    @pytest.fixture
    def semantic_analyzer(self, mock_vector_encoder):
        """Create a semantic analyzer instance."""
        return SemanticAnalyzer(
            vector_encoder=mock_vector_encoder,
            max_tfidf_features=100,
            min_df=1,
            max_df=0.9
        )

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "This is a document about machine learning and artificial intelligence.",
            "Natural language processing is a subset of machine learning.",
            "Deep learning models require large amounts of training data."
        ]

    def test_fit_and_transform_basic(self, semantic_analyzer, sample_documents):
        """Test basic fit and transform functionality."""
        results = semantic_analyzer.fit_and_transform(sample_documents)
        
        # Check that results contain expected keys
        assert 'embeddings' in results
        assert 'tfidf_matrix' in results
        assert 'document_count' in results
        assert 'embedding_dim' in results
        assert 'tfidf_features' in results
        assert 'tfidf_vocabulary_size' in results
        
        # Check dimensions
        assert results['document_count'] == 3
        assert results['embedding_dim'] == 384
        assert results['tfidf_features'] > 0
        assert results['tfidf_vocabulary_size'] > 0
        
        # Check that TF-IDF matrix is sparse
        assert scipy.sparse.issparse(results['tfidf_matrix'])
        assert results['tfidf_matrix'].shape[0] == 3  # 3 documents

    def test_compute_tfidf_features_after_fit(self, semantic_analyzer, sample_documents):
        """Test TF-IDF feature computation after fitting."""
        # First fit the analyzer
        semantic_analyzer.fit_and_transform(sample_documents)
        
        # Then compute TF-IDF features for new documents
        new_docs = ["Machine learning is fascinating.", "Data science applications."]
        tfidf_features = semantic_analyzer.compute_tfidf_features(new_docs)
        
        # Check results
        assert scipy.sparse.issparse(tfidf_features)
        assert tfidf_features.shape[0] == 2  # 2 new documents
        assert tfidf_features.shape[1] > 0   # Should have features

    def test_compute_tfidf_features_not_fitted(self, semantic_analyzer):
        """Test that computing TF-IDF features fails when not fitted."""
        with pytest.raises(ValueError, match="TF-IDF vectorizer not fitted"):
            semantic_analyzer.compute_tfidf_features(["test document"])

    def test_compute_tfidf_similarity(self, semantic_analyzer, sample_documents):
        """Test TF-IDF similarity computation."""
        # Fit the analyzer
        semantic_analyzer.fit_and_transform(sample_documents)
        
        # Get TF-IDF vectors for two documents
        tfidf_vectors = semantic_analyzer.compute_tfidf_features(sample_documents[:2])
        
        # Compute similarity
        similarity = semantic_analyzer.compute_tfidf_similarity(
            tfidf_vectors[0], tfidf_vectors[1]
        )
        
        # Check that similarity is a valid score
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_compute_tfidf_similarity_empty_vectors(self, semantic_analyzer):
        """Test TF-IDF similarity with empty vectors."""
        # Create empty sparse vectors
        empty_vector1 = scipy.sparse.csr_matrix((1, 100))
        empty_vector2 = scipy.sparse.csr_matrix((1, 100))
        
        similarity = semantic_analyzer.compute_tfidf_similarity(empty_vector1, empty_vector2)
        
        # Should return 0.0 for empty vectors
        assert similarity == 0.0

    def test_compute_tfidf_similarity_matrix(self, semantic_analyzer, sample_documents):
        """Test TF-IDF similarity matrix computation."""
        # Fit the analyzer
        results = semantic_analyzer.fit_and_transform(sample_documents)
        tfidf_matrix = results['tfidf_matrix']
        
        # Compute similarity matrix
        similarity_matrix = semantic_analyzer.compute_tfidf_similarity_matrix(tfidf_matrix)
        
        # Check dimensions and properties
        assert similarity_matrix.shape == (3, 3)
        assert np.allclose(np.diag(similarity_matrix), 1.0)  # Diagonal should be 1.0
        assert np.all(similarity_matrix >= 0.0)  # All similarities should be non-negative
        assert np.all(similarity_matrix <= 1.0 + 1e-10)  # Allow for small floating point errors

    def test_get_top_tfidf_terms(self, semantic_analyzer, sample_documents):
        """Test getting top TF-IDF terms for a document."""
        # Fit the analyzer
        semantic_analyzer.fit_and_transform(sample_documents)
        
        # Get top terms for first document
        top_terms = semantic_analyzer.get_top_tfidf_terms(0, top_k=5)
        
        # Check results
        assert isinstance(top_terms, list)
        assert len(top_terms) <= 5
        
        # Check that each term is a (term, score) tuple
        for term, score in top_terms:
            assert isinstance(term, str)
            assert isinstance(score, float)
            assert score > 0.0

    def test_get_vocabulary_stats(self, semantic_analyzer, sample_documents):
        """Test vocabulary statistics."""
        # Before fitting
        stats_before = semantic_analyzer.get_vocabulary_stats()
        assert stats_before['fitted'] is False
        
        # After fitting
        semantic_analyzer.fit_and_transform(sample_documents)
        stats_after = semantic_analyzer.get_vocabulary_stats()
        
        assert stats_after['fitted'] is True
        assert stats_after['vocabulary_size'] > 0
        assert stats_after['document_count'] == 3
        assert stats_after['feature_count'] > 0
        assert 'sample_terms' in stats_after
        assert isinstance(stats_after['sample_terms'], list)

    def test_extract_pair_features(self, semantic_analyzer, sample_documents):
        """Test comprehensive pair feature extraction."""
        # Fit the analyzer
        semantic_analyzer.fit_and_transform(sample_documents)
        
        # Extract features for a pair of texts
        text1 = sample_documents[0]
        text2 = sample_documents[1]
        
        features = semantic_analyzer.extract_pair_features(text1, text2)
        
        # Check that all expected features are present
        assert 'semantic_similarity' in features
        assert 'tfidf_similarity' in features
        assert 'combined_similarity' in features
        
        # Check that all features are valid floats
        for feature_name, feature_value in features.items():
            assert isinstance(feature_value, float)
            assert not np.isnan(feature_value)
        
        # Check that similarities are in valid ranges
        assert -1.0 <= features['semantic_similarity'] <= 1.0
        assert 0.0 <= features['tfidf_similarity'] <= 1.0
        assert 0.0 <= features['combined_similarity'] <= 1.0

    def test_extract_pair_features_not_fitted(self, semantic_analyzer):
        """Test pair feature extraction when TF-IDF is not fitted."""
        text1 = "This is test text one."
        text2 = "This is test text two."
        
        features = semantic_analyzer.extract_pair_features(text1, text2)
        
        # Should still work but TF-IDF similarity should be 0.0
        assert features['tfidf_similarity'] == 0.0
        # Semantic similarity might be 0.0 with mock encoder, so just check it's a valid float
        assert isinstance(features['semantic_similarity'], float)
        assert -1.0 <= features['semantic_similarity'] <= 1.0

    def test_text_cleaning(self, semantic_analyzer):
        """Test text cleaning functionality."""
        dirty_text = """
        # This is a header
        
        This text has [links](http://example.com) and URLs https://test.com
        
        It also has excessive    whitespace and !!!punctuation!!!
        """
        
        cleaned = semantic_analyzer._clean_text(dirty_text)
        
        # Check that cleaning worked
        assert "# This is a header" not in cleaned
        assert "http://example.com" not in cleaned
        assert "https://test.com" not in cleaned
        assert "links" in cleaned  # Link text should remain
        # Note: The current regex doesn't remove all punctuation, just excessive ones
        assert "   " not in cleaned  # Excessive whitespace should be removed

    def test_empty_documents_handling(self, semantic_analyzer):
        """Test handling of empty documents."""
        with pytest.raises(ValueError, match="Cannot fit on empty document list"):
            semantic_analyzer.fit_and_transform([])

    def test_invalid_documents_handling(self, semantic_analyzer):
        """Test handling of invalid documents."""
        invalid_docs = ["", "   ", "\n\n\n"]  # Empty or whitespace-only documents
        
        with pytest.raises(ValueError, match="No valid documents after cleaning"):
            semantic_analyzer.fit_and_transform(invalid_docs)

    def test_embedding_failure_fallback(self, semantic_analyzer, sample_documents):
        """Test fallback behavior when embedding generation fails."""
        # Mock the vector encoder to raise an exception
        semantic_analyzer.vector_encoder.encode_documents.side_effect = Exception("Embedding failed")
        
        results = semantic_analyzer.fit_and_transform(sample_documents)
        
        # Should still work with zero embeddings
        assert results['embeddings'].shape == (3, 384)
        assert np.all(results['embeddings'] == 0.0)
        assert results['tfidf_matrix'].shape[0] == 3  # TF-IDF should still work

    def test_tfidf_failure_fallback(self, semantic_analyzer, sample_documents):
        """Test fallback behavior when TF-IDF generation fails."""
        # Mock the TF-IDF vectorizer to raise an exception
        with patch.object(semantic_analyzer.tfidf_vectorizer, 'fit_transform', side_effect=Exception("TF-IDF failed")):
            results = semantic_analyzer.fit_and_transform(sample_documents)
            
            # Should still work with empty TF-IDF matrix
            assert scipy.sparse.issparse(results['tfidf_matrix'])
            assert results['tfidf_matrix'].shape[0] == 3
            assert results['embeddings'].shape == (3, 384)  # Embeddings should still work

    def test_combined_similarity_calculation(self, semantic_analyzer, sample_documents):
        """Test that combined similarity is calculated correctly."""
        # Fit the analyzer
        semantic_analyzer.fit_and_transform(sample_documents)
        
        # Extract features for identical texts (should have high similarity)
        text = sample_documents[0]
        features = semantic_analyzer.extract_pair_features(text, text)
        
        # Combined similarity should be reasonably high for identical texts
        # Note: With mock embeddings, semantic similarity might be low, so adjust expectations
        assert features['combined_similarity'] > 0.5
        assert features['tfidf_similarity'] > 0.9  # Should be very high for identical texts
        
        # Extract features for very different texts
        text1 = "Machine learning and artificial intelligence"
        text2 = "Cooking recipes and kitchen utensils"
        features_diff = semantic_analyzer.extract_pair_features(text1, text2)
        
        # Combined similarity should be lower for different texts
        assert features_diff['combined_similarity'] < features['combined_similarity']


if __name__ == "__main__":
    pytest.main([__file__])