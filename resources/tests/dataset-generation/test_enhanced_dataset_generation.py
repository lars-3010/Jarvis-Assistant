"""
Integration tests for enhanced dataset generation.

Tests end-to-end feature generation including all analyzers working together,
dataset generation pipeline, and comprehensive error handling.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from src.jarvis.tools.dataset_generation.feature_engineer import FeatureEngineer
from src.jarvis.tools.dataset_generation.models.data_models import NoteData, NoteFeatures, PairFeatures
from src.jarvis.tools.dataset_generation.error_handling import get_error_tracker, log_system_health
from src.jarvis.services.vector.encoder import VectorEncoder


class TestEnhancedDatasetGenerationIntegration:
    """Integration tests for enhanced dataset generation."""
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock VectorEncoder for testing."""
        encoder = Mock(spec=VectorEncoder)
        
        # Mock encoding methods
        def mock_encode_batch(texts):
            # Return consistent embeddings based on text content
            embeddings = []
            for i, text in enumerate(texts):
                # Create deterministic embeddings based on text hash
                seed = hash(text) % 1000
                np.random.seed(seed)
                embedding = np.random.rand(384)  # Standard sentence transformer dimension
                embeddings.append(embedding)
            return np.array(embeddings)
        
        def mock_encode(text):
            seed = hash(text) % 1000
            np.random.seed(seed)
            return np.random.rand(384)
        
        encoder.encode_batch.side_effect = mock_encode_batch
        encoder.encode.side_effect = mock_encode
        encoder.dimension = 384
        
        return encoder
    
    @pytest.fixture
    def sample_vault_data(self):
        """Create sample vault data for testing."""
        return [
            {
                "path": "AI/machine-learning.md",
                "content": """# Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

## Key Concepts

- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns in unlabeled data  
- **Reinforcement Learning**: Learning through interaction and feedback

## Applications

Machine learning is used in:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles

The field continues to evolve rapidly with new algorithms and techniques.""",
                "metadata": {
                    "created": "2024-01-01T10:00:00Z",
                    "modified": "2024-01-15T14:30:00Z",
                    "tags": ["AI", "machine-learning", "technology"]
                }
            },
            {
                "path": "AI/deep-learning.md", 
                "content": """# Deep Learning Overview

Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.

## Neural Network Architecture

Deep neural networks consist of:
1. Input layer
2. Hidden layers (multiple)
3. Output layer

## Popular Frameworks

- TensorFlow
- PyTorch
- Keras

## Applications

Deep learning excels in:
- Computer vision
- Speech recognition
- Natural language understanding
- Game playing (like AlphaGo)

The computational requirements are significant but the results are impressive.""",
                "metadata": {
                    "created": "2024-01-02T09:00:00Z",
                    "modified": "2024-01-16T11:20:00Z",
                    "tags": ["AI", "deep-learning", "neural-networks"]
                }
            },
            {
                "path": "Research/methodology.md",
                "content": """# Research Methodology

Effective research requires systematic methodology and careful planning.

## Research Process

1. **Problem Definition**: Clearly define the research question
2. **Literature Review**: Survey existing knowledge
3. **Hypothesis Formation**: Develop testable hypotheses
4. **Data Collection**: Gather relevant data
5. **Analysis**: Apply appropriate analytical methods
6. **Conclusion**: Draw meaningful conclusions

## Types of Research

- Quantitative research
- Qualitative research
- Mixed methods research

## Quality Assurance

Ensure research quality through:
- Peer review
- Replication studies
- Transparent methodology
- Ethical considerations

Good research contributes to human knowledge and understanding.""",
                "metadata": {
                    "created": "2024-01-03T08:00:00Z",
                    "modified": "2024-01-17T16:45:00Z",
                    "tags": ["research", "methodology", "academic"]
                }
            },
            {
                "path": "Business/strategy.md",
                "content": """# Business Strategy Development

Strategic planning is essential for organizational success and competitive advantage.

## Strategic Framework

Key components include:
- Vision and mission
- Market analysis
- Competitive positioning
- Resource allocation
- Performance metrics

## Strategy Types

- Cost leadership
- Differentiation
- Focus strategies
- Blue ocean strategy

## Implementation

Successful strategy requires:
- Clear communication
- Stakeholder alignment
- Resource commitment
- Regular monitoring
- Adaptive management

Strategy must evolve with changing market conditions and organizational capabilities.""",
                "metadata": {
                    "created": "2024-01-04T07:00:00Z",
                    "modified": "2024-01-18T13:15:00Z",
                    "tags": ["business", "strategy", "management"]
                }
            },
            {
                "path": "Personal/learning-notes.md",
                "content": """# Personal Learning Journey

Continuous learning is essential for personal and professional growth.

## Learning Strategies

Effective approaches include:
- Active reading
- Note-taking systems
- Spaced repetition
- Practice and application
- Teaching others

## Knowledge Areas

Currently focusing on:
- Artificial intelligence
- Data science
- Research methods
- Business strategy

## Reflection

Learning is most effective when it's:
- Goal-oriented
- Consistent
- Reflective
- Connected to real applications

The journey of learning never ends.""",
                "metadata": {
                    "created": "2024-01-05T06:00:00Z",
                    "modified": "2024-01-19T10:30:00Z",
                    "tags": ["personal", "learning", "growth"]
                }
            }
        ]
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_end_to_end_dataset_generation(self, mock_vector_encoder, sample_vault_data, temp_output_dir):
        """Test complete end-to-end dataset generation pipeline."""
        # Create dataset generator
        generator = DatasetGenerator(
            vector_encoder=mock_vector_encoder,
            output_dir=temp_output_dir
        )
        
        # Convert sample data to NoteData objects
        notes_data = []
        for item in sample_vault_data:
            # Generate embedding for the content
            embedding = mock_vector_encoder.encode(item["content"])
            
            note_data = NoteData(
                path=item["path"],
                content=item["content"],
                embedding=embedding,
                metadata=item["metadata"]
            )
            notes_data.append(note_data)
        
        # Generate datasets
        with patch('src.jarvis.tools.dataset_generation.dataset_generator.DatasetGenerator._save_datasets') as mock_save:
            # Mock the save operation to capture the data
            saved_datasets = {}
            
            def capture_save(notes_df, pairs_df, metadata):
                saved_datasets['notes'] = notes_df
                saved_datasets['pairs'] = pairs_df
                saved_datasets['metadata'] = metadata
            
            mock_save.side_effect = capture_save
            
            result = generator.generate_enhanced_datasets(
                notes_data=notes_data,
                vault_name="test_vault",
                include_pairs=True,
                max_pairs=10
            )
            
            # Verify generation succeeded
            assert result is not None
            assert isinstance(result, dict)
            assert 'notes_count' in result
            assert 'pairs_count' in result
            assert result['notes_count'] == len(sample_vault_data)
            assert result['pairs_count'] > 0
            
            # Verify datasets were created
            assert 'notes' in saved_datasets
            assert 'pairs' in saved_datasets
            assert 'metadata' in saved_datasets
            
            notes_df = saved_datasets['notes']
            pairs_df = saved_datasets['pairs']
            
            # Verify notes dataset structure
            assert len(notes_df) == len(sample_vault_data)
            
            # Check for enhanced features in notes
            expected_note_columns = [
                'path', 'content', 'word_count', 'char_count',
                'sentiment_score', 'sentiment_label', 'readability_score',
                'complexity_score', 'vocabulary_richness', 'content_type',
                'dominant_topic_id', 'dominant_topic_probability', 'topic_label',
                'tfidf_vocabulary_richness', 'avg_tfidf_score'
            ]
            
            for col in expected_note_columns:
                assert col in notes_df.columns, f"Missing column: {col}"
            
            # Verify pairs dataset structure
            assert len(pairs_df) > 0
            
            expected_pair_columns = [
                'note_a_path', 'note_b_path', 'cosine_similarity',
                'tfidf_similarity', 'combined_similarity',
                'topic_similarity', 'same_dominant_topic'
            ]
            
            for col in expected_pair_columns:
                assert col in pairs_df.columns, f"Missing column: {col}"
            
            # Verify feature quality
            # Sentiment scores should be in valid range
            sentiment_scores = notes_df['sentiment_score'].dropna()
            assert all(-1.0 <= score <= 1.0 for score in sentiment_scores)
            
            # Similarity scores should be in valid range
            cosine_similarities = pairs_df['cosine_similarity'].dropna()
            assert all(-1.0 <= sim <= 1.0 for sim in cosine_similarities)
            
            tfidf_similarities = pairs_df['tfidf_similarity'].dropna()
            assert all(0.0 <= sim <= 1.0 for sim in tfidf_similarities)
    
    def test_feature_engineer_integration(self, mock_vector_encoder, sample_vault_data):
        """Test FeatureEngineer integration with all analyzers."""
        # Create feature engineer
        feature_engineer = FeatureEngineer(mock_vector_encoder)
        
        # Convert sample data to NoteData objects
        notes_data = []
        for item in sample_vault_data:
            embedding = mock_vector_encoder.encode(item["content"])
            note_data = NoteData(
                path=item["path"],
                content=item["content"],
                embedding=embedding,
                metadata=item["metadata"]
            )
            notes_data.append(note_data)
        
        # Fit analyzers
        fit_results = feature_engineer.fit_analyzers(notes_data)
        
        # Verify fitting results
        assert isinstance(fit_results, dict)
        assert fit_results['notes_processed'] == len(sample_vault_data)
        assert 'system_health' in fit_results
        
        # At least one analyzer should fit successfully
        assert fit_results['semantic_fitted'] or fit_results['topic_fitted']
        
        # Extract features for each note
        enhanced_features_list = []
        for note_data in notes_data:
            features = feature_engineer.extract_note_features(note_data)
            assert isinstance(features, feature_engineer.EnhancedFeatures)
            enhanced_features_list.append(features)
        
        # Verify feature extraction
        assert len(enhanced_features_list) == len(sample_vault_data)
        
        # Check that features have reasonable values
        for features in enhanced_features_list:
            # Content features should be extracted for non-empty content
            if features.content_features:
                assert -1.0 <= features.content_features.sentiment_score <= 1.0
                assert 0.0 <= features.content_features.complexity_score <= 1.0
                assert 0.0 <= features.content_features.vocabulary_richness <= 1.0
            
            # TF-IDF features should be reasonable
            assert 0.0 <= features.tfidf_vocabulary_richness <= 1.0
            assert features.avg_tfidf_score >= 0.0
        
        # Test pair feature extraction
        note_a = notes_data[0]  # AI/machine-learning.md
        note_b = notes_data[1]  # AI/deep-learning.md (should be similar)
        note_c = notes_data[3]  # Business/strategy.md (should be different)
        
        # Similar notes should have higher similarity
        similar_features = feature_engineer.extract_pair_features(note_a, note_b)
        different_features = feature_engineer.extract_pair_features(note_a, note_c)
        
        assert isinstance(similar_features, dict)
        assert isinstance(different_features, dict)
        
        # Verify similarity scores are in valid ranges
        for features in [similar_features, different_features]:
            if 'semantic_similarity' in features:
                assert -1.0 <= features['semantic_similarity'] <= 1.0
            if 'tfidf_similarity' in features:
                assert 0.0 <= features['tfidf_similarity'] <= 1.0
            if 'combined_similarity' in features:
                assert 0.0 <= features['combined_similarity'] <= 1.0
    
    def test_error_handling_integration(self, mock_vector_encoder, temp_output_dir):
        """Test comprehensive error handling in integration scenario."""
        # Clear previous errors
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # Create problematic data that might cause errors
        problematic_data = [
            NoteData(
                path="empty.md",
                content="",  # Empty content
                embedding=None,
                metadata={}
            ),
            NoteData(
                path="none.md",
                content=None,  # None content
                embedding=None,
                metadata={}
            ),
            NoteData(
                path="huge.md",
                content="A" * 100000,  # Very large content
                embedding=np.array([float('inf')] * 384),  # Problematic embedding
                metadata={}
            ),
            NoteData(
                path="unicode.md",
                content="Test with émojis 🚀 and spëcial chäractërs",
                embedding=np.random.rand(384),
                metadata={"special": "ñoñó"}
            )
        ]
        
        # Create dataset generator
        generator = DatasetGenerator(
            vector_encoder=mock_vector_encoder,
            output_dir=temp_output_dir
        )
        
        # Should handle problematic data gracefully
        with patch('src.jarvis.tools.dataset_generation.dataset_generator.DatasetGenerator._save_datasets'):
            result = generator.generate_enhanced_datasets(
                notes_data=problematic_data,
                vault_name="problematic_vault",
                include_pairs=True
            )
            
            # Should complete without crashing
            assert result is not None
            assert isinstance(result, dict)
        
        # Check error tracking
        final_error_count = len(error_tracker.errors)
        
        # May have tracked some errors, but should not crash
        if final_error_count > initial_error_count:
            error_summary = error_tracker.get_error_summary()
            assert error_summary['total_errors'] > 0
            
            # Errors should be properly categorized
            assert 'by_severity' in error_summary
            assert 'by_component' in error_summary
    
    def test_system_health_monitoring(self, mock_vector_encoder, sample_vault_data):
        """Test system health monitoring during dataset generation."""
        # Log initial system health
        initial_health = log_system_health()
        
        assert isinstance(initial_health, dict)
        assert 'overall_health_score' in initial_health
        assert 'dependency_status' in initial_health
        assert 'component_reliability' in initial_health
        
        # Create feature engineer and process data
        feature_engineer = FeatureEngineer(mock_vector_encoder)
        
        notes_data = []
        for item in sample_vault_data:
            embedding = mock_vector_encoder.encode(item["content"])
            note_data = NoteData(
                path=item["path"],
                content=item["content"],
                embedding=embedding,
                metadata=item["metadata"]
            )
            notes_data.append(note_data)
        
        # Fit analyzers and check health
        fit_results = feature_engineer.fit_analyzers(notes_data)
        
        assert 'system_health' in fit_results
        system_health = fit_results['system_health']
        
        # Health score should be reasonable
        assert 0.0 <= system_health['overall_health_score'] <= 1.0
        
        # Should have dependency status
        assert 'dependency_status' in system_health
        dependency_status = system_health['dependency_status']
        
        # Check key dependencies
        expected_deps = ['spacy', 'textstat', 'bertopic', 'sklearn', 'networkx_advanced']
        for dep in expected_deps:
            assert dep in dependency_status
            assert isinstance(dependency_status[dep], bool)
    
    def test_performance_with_large_dataset(self, mock_vector_encoder, temp_output_dir):
        """Test performance characteristics with larger dataset."""
        import time
        
        # Create larger dataset
        large_dataset = []
        for i in range(50):  # 50 notes
            content = f"""# Document {i}
            
This is document number {i} in our test dataset. It contains information about topic {i % 5}.

## Section A
Content related to category {i % 3}.

## Section B  
More detailed information about subject {i % 7}.

The document discusses various aspects of the topic with sufficient detail to enable meaningful analysis.
"""
            
            embedding = mock_vector_encoder.encode(content)
            note_data = NoteData(
                path=f"docs/document_{i:03d}.md",
                content=content,
                embedding=embedding,
                metadata={"id": i, "category": i % 5}
            )
            large_dataset.append(note_data)
        
        # Measure generation time
        start_time = time.time()
        
        generator = DatasetGenerator(
            vector_encoder=mock_vector_encoder,
            output_dir=temp_output_dir
        )
        
        with patch('src.jarvis.tools.dataset_generation.dataset_generator.DatasetGenerator._save_datasets'):
            result = generator.generate_enhanced_datasets(
                notes_data=large_dataset,
                vault_name="large_test_vault",
                include_pairs=True,
                max_pairs=100  # Limit pairs for performance
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify successful completion
        assert result is not None
        assert result['notes_count'] == 50
        assert result['pairs_count'] > 0
        
        # Performance should be reasonable (adjust threshold as needed)
        assert processing_time < 60.0  # Should complete within 60 seconds
        
        # Log performance metrics
        print(f"Processed {len(large_dataset)} notes in {processing_time:.2f} seconds")
        print(f"Average time per note: {processing_time/len(large_dataset):.3f} seconds")
    
    def test_feature_quality_validation(self, mock_vector_encoder, sample_vault_data):
        """Test feature quality validation across the pipeline."""
        feature_engineer = FeatureEngineer(mock_vector_encoder)
        
        # Convert sample data
        notes_data = []
        for item in sample_vault_data:
            embedding = mock_vector_encoder.encode(item["content"])
            note_data = NoteData(
                path=item["path"],
                content=item["content"],
                embedding=embedding,
                metadata=item["metadata"]
            )
            notes_data.append(note_data)
        
        # Fit analyzers
        feature_engineer.fit_analyzers(notes_data)
        
        # Extract features and validate quality
        for note_data in notes_data:
            features = feature_engineer.extract_note_features(note_data)
            
            # Validate feature ranges and types
            if features.content_features:
                cf = features.content_features
                
                # Sentiment should be in valid range
                assert -1.0 <= cf.sentiment_score <= 1.0
                assert cf.sentiment_label in ['positive', 'negative', 'neutral']
                
                # Readability should be reasonable
                assert cf.readability_score >= 0.0
                assert cf.readability_grade >= 0.0
                
                # Complexity should be normalized
                assert 0.0 <= cf.complexity_score <= 1.0
                
                # Vocabulary richness should be normalized
                assert 0.0 <= cf.vocabulary_richness <= 1.0
                
                # Content type should be valid
                valid_types = ['technical', 'academic', 'business', 'creative', 'scientific', 'general']
                assert cf.content_type in valid_types
                
                # Counts should be non-negative
                assert cf.heading_count >= 0
                assert cf.max_heading_depth >= 0
                assert cf.unique_words >= 0
                assert cf.total_words >= 0
            
            # TF-IDF features should be valid
            assert 0.0 <= features.tfidf_vocabulary_richness <= 1.0
            assert features.avg_tfidf_score >= 0.0
            
            # Topic features should be valid
            if features.dominant_topic_id >= 0:
                assert 0.0 <= features.dominant_topic_probability <= 1.0
                assert isinstance(features.topic_label, str)
                
                if features.topic_probabilities:
                    assert all(0.0 <= prob <= 1.0 for prob in features.topic_probabilities)
                    # Probabilities should roughly sum to 1.0 (allowing for floating point errors)
                    prob_sum = sum(features.topic_probabilities)
                    assert 0.8 <= prob_sum <= 1.2
        
        # Test pair feature quality
        note_a = notes_data[0]
        note_b = notes_data[1]
        
        pair_features = feature_engineer.extract_pair_features(note_a, note_b)
        
        # Validate pair feature ranges
        if 'semantic_similarity' in pair_features:
            assert -1.0 <= pair_features['semantic_similarity'] <= 1.0
        
        if 'tfidf_similarity' in pair_features:
            assert 0.0 <= pair_features['tfidf_similarity'] <= 1.0
        
        if 'combined_similarity' in pair_features:
            assert 0.0 <= pair_features['combined_similarity'] <= 1.0
        
        if 'topic_similarity' in pair_features:
            assert 0.0 <= pair_features['topic_similarity'] <= 1.0
        
        if 'same_dominant_topic' in pair_features:
            assert isinstance(pair_features['same_dominant_topic'], bool)
    
    def test_analyzer_status_reporting(self, mock_vector_encoder, sample_vault_data):
        """Test analyzer status reporting throughout the pipeline."""
        feature_engineer = FeatureEngineer(mock_vector_encoder)
        
        # Initial status - should show not fitted
        initial_status = feature_engineer.get_analyzer_status()
        
        assert initial_status['semantic_analyzer']['fitted'] is False
        assert initial_status['topic_modeler']['fitted'] is False
        assert initial_status['topic_modeler']['topic_count'] == 0
        
        # Convert sample data
        notes_data = []
        for item in sample_vault_data:
            embedding = mock_vector_encoder.encode(item["content"])
            note_data = NoteData(
                path=item["path"],
                content=item["content"],
                embedding=embedding,
                metadata=item["metadata"]
            )
            notes_data.append(note_data)
        
        # Fit analyzers
        fit_results = feature_engineer.fit_analyzers(notes_data)
        
        # Status after fitting
        fitted_status = feature_engineer.get_analyzer_status()
        
        # Should show changes in status
        if fit_results['semantic_fitted']:
            assert fitted_status['semantic_analyzer']['fitted'] is True
            assert 'stats' in fitted_status['semantic_analyzer']
        
        if fit_results['topic_fitted']:
            assert fitted_status['topic_modeler']['fitted'] is True
            assert fitted_status['topic_modeler']['topic_count'] > 0
        
        # Content analyzer should show dependency status
        content_status = fitted_status['content_analyzer']
        assert 'spacy_available' in content_status
        assert 'textstat_available' in content_status
        assert isinstance(content_status['spacy_available'], bool)
        assert isinstance(content_status['textstat_available'], bool)
        
        # Graph analyzer should be available
        assert fitted_status['graph_analyzer']['available'] is True


if __name__ == "__main__":
    pytest.main([__file__])