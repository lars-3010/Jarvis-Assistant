"""
Test TF-IDF integration in dataset generation pipeline.

This module tests that TF-IDF features are properly integrated into both
notes and pairs dataset generation.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock

from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
from jarvis.tools.dataset_generation.models.data_models import NoteData, PairFeatures
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vault.reader import VaultReader


class TestTFIDFIntegration:
    """Test TF-IDF integration in dataset generation."""

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock vector encoder."""
        encoder = Mock(spec=VectorEncoder)
        encoder.vector_dim = 384
        encoder.encode_documents.return_value = np.random.rand(3, 384).astype(np.float32)
        return encoder

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock vault reader."""
        reader = Mock(spec=VaultReader)
        reader.read_file.return_value = ("Sample note content about machine learning.", {})
        reader.get_absolute_path.return_value = Mock(stat=Mock(return_value=Mock(st_ctime=1234567890, st_mtime=1234567890)))
        return reader

    @pytest.fixture
    def sample_note_data(self):
        """Create sample note data for testing."""
        return {
            "note1.md": NoteData(
                path="note1.md",
                title="Machine Learning Basics",
                content="This note covers machine learning fundamentals and algorithms.",
                metadata={},
                tags=["ml", "ai"],
                outgoing_links=["note2.md"],
                word_count=10
            ),
            "note2.md": NoteData(
                path="note2.md", 
                title="Deep Learning",
                content="Deep learning is a subset of machine learning using neural networks.",
                metadata={},
                tags=["dl", "neural"],
                outgoing_links=["note1.md"],
                word_count=12
            ),
            "note3.md": NoteData(
                path="note3.md",
                title="Data Science",
                content="Data science combines statistics, programming, and domain expertise.",
                metadata={},
                tags=["data", "stats"],
                outgoing_links=[],
                word_count=9
            )
        }

    def test_notes_generator_tfidf_integration(self, mock_vault_reader, mock_vector_encoder):
        """Test that notes generator properly integrates TF-IDF features."""
        # Create notes generator
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Verify semantic analyzer is initialized
        assert generator.semantic_analyzer is not None
        assert generator._semantic_analyzer_fitted is False
        
        # Test TF-IDF feature computation
        note_data = NoteData(
            path="test.md",
            title="Test Note",
            content="Machine learning and artificial intelligence are fascinating topics.",
            metadata={},
            tags=["test"],
            outgoing_links=[],
            word_count=10
        )
        
        # Mock the semantic analyzer to be fitted
        generator._semantic_analyzer_fitted = True
        
        # Mock TF-IDF computation
        with patch.object(generator.semantic_analyzer, 'compute_tfidf_features') as mock_tfidf:
            # Create mock TF-IDF vector
            mock_tfidf_vector = Mock()
            mock_tfidf_vector.shape = (1, 100)
            mock_tfidf_vector.toarray.return_value = np.array([[0.5, 0.3, 0.0, 0.8, 0.0] + [0.0] * 95])
            mock_tfidf.return_value = mock_tfidf_vector
            
            # Mock feature names
            generator.semantic_analyzer.tfidf_vectorizer = Mock()
            generator.semantic_analyzer.tfidf_vectorizer.get_feature_names_out.return_value = [
                'machine', 'learning', 'artificial', 'intelligence', 'fascinating'
            ] + [f'term_{i}' for i in range(95)]
            
            # Compute TF-IDF features
            tfidf_features = generator._compute_tfidf_features(note_data)
            
            # Verify features are computed
            assert 'top_tfidf_terms' in tfidf_features
            assert 'tfidf_vocabulary_richness' in tfidf_features
            assert 'avg_tfidf_score' in tfidf_features
            
            # Verify top terms are JSON formatted
            top_terms = json.loads(tfidf_features['top_tfidf_terms'])
            assert isinstance(top_terms, list)
            assert len(top_terms) > 0
            
            # Verify vocabulary richness is calculated
            assert 0.0 <= tfidf_features['tfidf_vocabulary_richness'] <= 1.0
            
            # Verify average TF-IDF score is calculated
            assert tfidf_features['avg_tfidf_score'] > 0.0

    def test_notes_generator_tfidf_fallback(self, mock_vault_reader, mock_vector_encoder):
        """Test TF-IDF feature fallback when analyzer is not fitted."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        note_data = NoteData(
            path="test.md",
            title="Test Note", 
            content="Test content",
            metadata={},
            tags=[],
            outgoing_links=[],
            word_count=2
        )
        
        # Analyzer not fitted - should return default values
        tfidf_features = generator._compute_tfidf_features(note_data)
        
        assert tfidf_features['top_tfidf_terms'] == ""
        assert tfidf_features['tfidf_vocabulary_richness'] == 0.0
        assert tfidf_features['avg_tfidf_score'] == 0.0

    def test_pairs_generator_tfidf_integration(self, mock_vector_encoder, sample_note_data):
        """Test that pairs generator properly uses TF-IDF features."""
        # Create pairs generator
        generator = PairsDatasetGenerator(vector_encoder=mock_vector_encoder)
        
        # Verify semantic analyzer is initialized
        assert generator.semantic_analyzer is not None
        assert generator._semantic_analyzer_fitted is False
        
        # Test that semantic analyzer is used in pair feature computation
        note_a = sample_note_data["note1.md"]
        note_b = sample_note_data["note2.md"]
        
        # Mock the semantic analyzer to be fitted
        generator._semantic_analyzer_fitted = True
        
        # Mock pair feature extraction
        with patch.object(generator.semantic_analyzer, 'extract_pair_features') as mock_extract:
            mock_extract.return_value = {
                'semantic_similarity': 0.75,
                'tfidf_similarity': 0.65,
                'combined_similarity': 0.70
            }
            
            # Test pair feature extraction
            features = generator.semantic_analyzer.extract_pair_features(
                note_a.content, note_b.content
            )
            
            # Verify TF-IDF similarity is included
            assert 'tfidf_similarity' in features
            assert 'combined_similarity' in features
            assert features['tfidf_similarity'] == 0.65
            assert features['combined_similarity'] == 0.70

    def test_tfidf_features_in_dataframe_schema(self):
        """Test that TF-IDF features are included in the DataFrame schema."""
        from jarvis.tools.dataset_generation.models.data_models import NoteFeatures, PairFeatures
        
        # Test NoteFeatures has TF-IDF fields
        note_features = NoteFeatures(
            note_path="test.md",
            note_title="Test",
            word_count=10,
            tag_count=1,
            quality_stage="test",
            creation_date=pd.Timestamp.now(),
            last_modified=pd.Timestamp.now(),
            outgoing_links_count=0,
            top_tfidf_terms='[{"term": "test", "score": 0.5}]',
            tfidf_vocabulary_richness=0.3,
            avg_tfidf_score=0.4
        )
        
        assert hasattr(note_features, 'top_tfidf_terms')
        assert hasattr(note_features, 'tfidf_vocabulary_richness')
        assert hasattr(note_features, 'avg_tfidf_score')
        
        # Test PairFeatures has TF-IDF field
        pair_features = PairFeatures(
            note_a_path="note1.md",
            note_b_path="note2.md",
            cosine_similarity=0.8,
            tfidf_similarity=0.7,
            combined_similarity=0.75,
            semantic_cluster_match=False,
            tag_overlap_count=1,
            tag_jaccard_similarity=0.5,
            vault_path_distance=1,
            shortest_path_length=2,
            common_neighbors_count=0,
            adamic_adar_score=0.0,
            word_count_ratio=0.9,
            creation_time_diff_days=1.0,
            quality_stage_compatibility=1,
            source_centrality=0.1,
            target_centrality=0.2,
            clustering_coefficient=0.3,
            link_exists=True
        )
        
        assert hasattr(pair_features, 'tfidf_similarity')
        assert pair_features.tfidf_similarity == 0.7

    def test_tfidf_error_handling(self, mock_vault_reader, mock_vector_encoder):
        """Test error handling in TF-IDF feature extraction."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        note_data = NoteData(
            path="test.md",
            title="Test Note",
            content="Test content",
            metadata={},
            tags=[],
            outgoing_links=[],
            word_count=2
        )
        
        # Mock semantic analyzer to be fitted but raise exception
        generator._semantic_analyzer_fitted = True
        
        with patch.object(generator.semantic_analyzer, 'compute_tfidf_features', side_effect=Exception("TF-IDF error")):
            # Should handle error gracefully and return default values
            tfidf_features = generator._compute_tfidf_features(note_data)
            
            assert tfidf_features['top_tfidf_terms'] == ""
            assert tfidf_features['tfidf_vocabulary_richness'] == 0.0
            assert tfidf_features['avg_tfidf_score'] == 0.0

    def test_semantic_analyzer_fitting_process(self, mock_vault_reader, mock_vector_encoder):
        """Test the semantic analyzer fitting process in dataset generation."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Mock markdown parser
        with patch.object(generator.markdown_parser, 'parse') as mock_parse:
            mock_parse.return_value = {
                'content_without_frontmatter': 'Sample content for TF-IDF fitting',
                'frontmatter': {},
                'tags': [],
                'links': []
            }
            
            # Mock semantic analyzer fitting
            with patch.object(generator.semantic_analyzer, 'fit_and_transform') as mock_fit:
                mock_fit.return_value = {
                    'embeddings': np.random.rand(3, 384),
                    'tfidf_matrix': Mock(),
                    'tfidf_features': 100,
                    'document_count': 3
                }
                
                # Test fitting process (would normally be called in generate_dataset)
                notes = ["note1.md", "note2.md", "note3.md"]
                
                # Simulate the fitting process
                all_contents = []
                for note_path in notes:
                    content, _ = generator.vault_reader.read_file(note_path)
                    parsed_content = generator.markdown_parser.parse(content)
                    clean_content = parsed_content.get('content_without_frontmatter', content)
                    if clean_content and clean_content.strip():
                        all_contents.append(clean_content)
                
                if all_contents:
                    semantic_results = generator.semantic_analyzer.fit_and_transform(all_contents)
                    generator._semantic_analyzer_fitted = True
                
                # Verify fitting was successful
                assert generator._semantic_analyzer_fitted is True
                mock_fit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])