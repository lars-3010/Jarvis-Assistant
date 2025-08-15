"""
Test content analysis functionality.

This module tests the ContentAnalyzer class and its integration with the dataset
generation pipeline for comprehensive content analysis features.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from jarvis.tools.dataset_generation.analyzers.content_analyzer import ContentAnalyzer, ContentFeatures
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.models.data_models import NoteData
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vault.reader import VaultReader


class TestContentAnalyzer:
    """Test ContentAnalyzer functionality."""

    @pytest.fixture
    def content_analyzer(self):
        """Create a content analyzer instance."""
        return ContentAnalyzer(use_spacy=False, use_textstat=False)  # Disable external deps for testing

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        # Machine Learning Fundamentals
        
        Machine learning is a fascinating field that combines statistics, computer science, and domain expertise.
        
        ## Key Concepts
        
        - Supervised learning uses labeled data
        - Unsupervised learning finds patterns in unlabeled data
        - Deep learning uses neural networks
        
        ```python
        def train_model(data):
            return model.fit(data)
        ```
        
        This is an excellent approach for solving complex problems!
        """

    def test_content_analyzer_initialization(self):
        """Test ContentAnalyzer initialization."""
        # Test with external dependencies disabled
        analyzer = ContentAnalyzer(use_spacy=False, use_textstat=False)
        assert analyzer.use_spacy is False
        assert analyzer.use_textstat is False
        assert analyzer._spacy_available is False
        assert analyzer._textstat_available is False

    def test_analyze_content_basic(self, content_analyzer, sample_text):
        """Test basic content analysis functionality."""
        features = content_analyzer.analyze_content(sample_text)
        
        # Check that we get a ContentFeatures object
        assert isinstance(features, ContentFeatures)
        
        # Check basic structure analysis
        assert features.heading_count > 0
        assert features.max_heading_depth > 0
        assert features.list_count > 0
        assert features.code_block_count > 0
        
        # Check vocabulary analysis
        assert features.total_words > 0
        assert features.unique_words > 0
        assert 0.0 <= features.vocabulary_richness <= 1.0
        
        # Check content type classification
        assert features.content_type in ['technical', 'academic', 'business', 'creative', 'scientific', 'general']
        
        # Check complexity score
        assert 0.0 <= features.complexity_score <= 1.0

    def test_analyze_empty_content(self, content_analyzer):
        """Test content analysis with empty content."""
        features = content_analyzer.analyze_content("")
        
        # Should return default ContentFeatures
        assert features.sentiment_score == 0.0
        assert features.sentiment_label == "neutral"
        assert features.readability_score == 0.0
        assert features.complexity_score == 0.0
        assert features.total_words == 0
        assert features.unique_words == 0

    def test_sentiment_analysis(self, content_analyzer):
        """Test sentiment analysis functionality."""
        # Positive text
        positive_text = "This is an excellent and wonderful approach! I love this amazing solution."
        features_pos = content_analyzer.analyze_content(positive_text)
        assert features_pos.sentiment_score > 0
        assert features_pos.sentiment_label == "positive"
        
        # Negative text
        negative_text = "This is a terrible and awful approach. I hate this horrible solution."
        features_neg = content_analyzer.analyze_content(negative_text)
        assert features_neg.sentiment_score < 0
        assert features_neg.sentiment_label == "negative"
        
        # Neutral text
        neutral_text = "This is a document about machine learning algorithms and data structures."
        features_neu = content_analyzer.analyze_content(neutral_text)
        assert features_neu.sentiment_label == "neutral"

    def test_readability_analysis_fallback(self, content_analyzer):
        """Test readability analysis with fallback calculations."""
        text = "This is a simple sentence. This is another simple sentence with more words."
        features = content_analyzer.analyze_content(text)
        
        # Should have some readability metrics even without textstat
        assert features.avg_sentence_length > 0
        assert features.avg_word_length > 0

    def test_structure_analysis(self, content_analyzer):
        """Test document structure analysis."""
        structured_text = """
        # Main Heading
        ## Sub Heading
        ### Deep Heading
        
        - List item 1
        - List item 2
        
        1. Numbered item 1
        2. Numbered item 2
        
        ```python
        code_block = True
        ```
        
        ```javascript
        another_block = true;
        ```
        """
        
        features = content_analyzer.analyze_content(structured_text)
        
        assert features.heading_count == 3
        assert features.max_heading_depth == 3
        assert features.list_count == 4  # 2 bullet + 2 numbered
        assert features.code_block_count == 2

    def test_vocabulary_analysis(self, content_analyzer):
        """Test vocabulary richness analysis."""
        # Text with high vocabulary richness
        rich_text = "The quick brown fox jumps over the lazy dog."
        features_rich = content_analyzer.analyze_content(rich_text)
        
        # Text with low vocabulary richness (repeated words)
        poor_text = "The the the fox fox fox jumps jumps jumps."
        features_poor = content_analyzer.analyze_content(poor_text)
        
        # Rich text should have higher vocabulary richness
        assert features_rich.vocabulary_richness > features_poor.vocabulary_richness
        assert features_rich.unique_words > features_poor.unique_words

    def test_technical_density_calculation(self, content_analyzer):
        """Test technical term density calculation."""
        # Technical text
        technical_text = "The Algorithm uses DataStructure and implements AbstractFactory pattern with CamelCase naming."
        features_tech = content_analyzer.analyze_content(technical_text)
        
        # Non-technical text
        simple_text = "The cat sat on the mat and looked at the sun."
        features_simple = content_analyzer.analyze_content(simple_text)
        
        # Technical text should have higher technical density
        assert features_tech.technical_density > features_simple.technical_density

    def test_concept_density_calculation(self, content_analyzer):
        """Test concept density calculation."""
        # Conceptual text
        conceptual_text = "The Theory of Relativity is a Concept that involves Analysis and Synthesis of Ideas."
        features_concept = content_analyzer.analyze_content(conceptual_text)
        
        # Simple text
        simple_text = "the cat sat on the mat and looked around."
        features_simple = content_analyzer.analyze_content(simple_text)
        
        # Conceptual text should have higher or equal concept density
        assert features_concept.concept_density >= features_simple.concept_density

    def test_content_type_classification(self, content_analyzer):
        """Test content type classification."""
        # Technical content
        tech_text = "This algorithm implements a function using object-oriented programming and software architecture."
        features_tech = content_analyzer.analyze_content(tech_text)
        assert features_tech.content_type == "technical"
        assert "algorithm" in features_tech.domain_indicators
        
        # Academic content
        academic_text = "This research study analyzes the methodology and presents findings from the literature review."
        features_academic = content_analyzer.analyze_content(academic_text)
        assert features_academic.content_type == "academic"
        assert "research" in features_academic.domain_indicators
        
        # Business content
        business_text = "The business strategy focuses on market analysis and customer revenue optimization."
        features_business = content_analyzer.analyze_content(business_text)
        assert features_business.content_type == "business"
        assert "business" in features_business.domain_indicators

    def test_complexity_score_calculation(self, content_analyzer):
        """Test complexity score calculation."""
        # Complex text
        complex_text = """
        The implementation of the AbstractFactoryPattern requires sophisticated understanding of 
        object-oriented design principles and architectural patterns. This methodology involves 
        comprehensive analysis of system requirements and synthesis of multiple design concepts.
        """
        features_complex = content_analyzer.analyze_content(complex_text)
        
        # Simple text
        simple_text = "The cat is big. The dog is small. They are pets."
        features_simple = content_analyzer.analyze_content(simple_text)
        
        # Complex text should have higher complexity score
        assert features_complex.complexity_score > features_simple.complexity_score

    def test_text_cleaning(self, content_analyzer):
        """Test text cleaning functionality."""
        dirty_text = "This has **bold** and *italic* and `code` and [links](http://example.com)."
        clean_text = content_analyzer._clean_text(dirty_text)
        
        # Should remove markdown formatting but keep text
        assert "**" not in clean_text
        assert "*" not in clean_text
        assert "`" not in clean_text
        assert "bold" in clean_text
        assert "italic" in clean_text
        assert "code" in clean_text
        assert "links" in clean_text

    def test_analysis_summary(self, content_analyzer, sample_text):
        """Test analysis summary generation."""
        features = content_analyzer.analyze_content(sample_text)
        summary = content_analyzer.get_analysis_summary(features)
        
        # Check that summary contains expected keys
        expected_keys = [
            'content_type', 'complexity_level', 'readability_level', 'sentiment',
            'technical_content', 'entity_count', 'structure_richness', 'vocabulary_richness'
        ]
        
        for key in expected_keys:
            assert key in summary
        
        # Check that values are reasonable
        assert summary['complexity_level'] in ['low', 'medium', 'high']
        assert summary['readability_level'] in ['easy', 'moderate', 'difficult']
        assert summary['sentiment'] in ['positive', 'negative', 'neutral']
        assert isinstance(summary['technical_content'], bool)
        assert isinstance(summary['entity_count'], int)
        assert summary['structure_richness'] in ['low', 'medium', 'high']
        assert summary['vocabulary_richness'] in ['low', 'medium', 'high']

    def test_spacy_integration_fallback(self, content_analyzer):
        """Test spaCy integration fallback when not available."""
        # spaCy not available - should return empty results
        entities_data = content_analyzer._extract_named_entities("Python is great")
        
        assert entities_data['entities'] == []
        assert entities_data['types'] == {}

    def test_textstat_integration_fallback(self, content_analyzer):
        """Test textstat integration fallback when not available."""
        # textstat not available - should use fallback calculations
        text = "This is a test sentence for readability analysis."
        metrics = content_analyzer._analyze_readability(text)
        
        # Should have some basic metrics even without textstat
        assert 'flesch_score' in metrics
        assert 'grade_level' in metrics
        assert 'avg_sentence_length' in metrics
        assert 'avg_word_length' in metrics


class TestContentAnalysisIntegration:
    """Test content analysis integration with dataset generation."""

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock vector encoder."""
        encoder = Mock(spec=VectorEncoder)
        encoder.vector_dim = 384
        encoder.encode_documents.return_value = [[0.1] * 384]
        return encoder

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock vault reader."""
        reader = Mock(spec=VaultReader)
        reader.read_file.return_value = ("Sample content for analysis.", {})
        reader.get_absolute_path.return_value = Mock(stat=Mock(return_value=Mock(st_ctime=1234567890, st_mtime=1234567890)))
        return reader

    def test_notes_generator_content_analysis_integration(self, mock_vault_reader, mock_vector_encoder):
        """Test that notes generator properly integrates content analysis."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Verify content analyzer is initialized
        assert generator.content_analyzer is not None
        assert isinstance(generator.content_analyzer, ContentAnalyzer)
        
        # Test content feature computation
        note_data = NoteData(
            path="test.md",
            title="Test Note",
            content="This is excellent technical content about algorithms and data structures!",
            metadata={},
            tags=["test"],
            outgoing_links=[],
            word_count=10
        )
        
        # Compute content features
        content_features = generator._compute_content_features(note_data)
        
        # Verify features are computed
        assert 'sentiment_score' in content_features
        assert 'sentiment_label' in content_features
        assert 'readability_score' in content_features
        assert 'content_type' in content_features
        assert 'complexity_score' in content_features
        
        # Verify sentiment analysis worked
        assert content_features['sentiment_score'] > 0  # Should be positive due to "excellent"
        assert content_features['sentiment_label'] == 'positive'
        
        # Verify content type classification (could be technical or scientific)
        assert content_features['content_type'] in ['technical', 'scientific']  # Should detect technical/scientific content

    def test_content_features_error_handling(self, mock_vault_reader, mock_vector_encoder):
        """Test error handling in content feature extraction."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Test with empty content
        note_data = NoteData(
            path="empty.md",
            title="Empty Note",
            content="",
            metadata={},
            tags=[],
            outgoing_links=[],
            word_count=0
        )
        
        content_features = generator._compute_content_features(note_data)
        
        # Should return default values for empty content
        assert content_features['sentiment_score'] == 0.0
        assert content_features['sentiment_label'] == 'neutral'
        assert content_features['readability_score'] == 0.0
        assert content_features['content_type'] == 'general'

    def test_content_features_json_serialization(self, mock_vault_reader, mock_vector_encoder):
        """Test JSON serialization of content features."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Mock content analyzer to return entities
        mock_features = Mock()
        mock_features.sentiment_score = 0.5
        mock_features.sentiment_label = 'positive'
        mock_features.readability_score = 75.0
        mock_features.readability_grade = 8.0
        mock_features.content_type = 'technical'
        mock_features.complexity_score = 0.6
        mock_features.vocabulary_richness = 0.7
        mock_features.named_entities = [
            {'text': 'Python', 'label': 'PRODUCT', 'description': 'Product'}
        ]
        mock_features.entity_types = {'PRODUCT': 1}
        
        with patch.object(generator.content_analyzer, 'analyze_content', return_value=mock_features):
            note_data = NoteData(
                path="test.md",
                title="Test Note",
                content="Python is a programming language.",
                metadata={},
                tags=[],
                outgoing_links=[],
                word_count=5
            )
            
            content_features = generator._compute_content_features(note_data)
            
            # Verify JSON serialization worked
            assert content_features['named_entities_json'] != ""
            assert content_features['entity_types_json'] != ""
            
            # Verify JSON can be parsed
            entities = json.loads(content_features['named_entities_json'])
            entity_types = json.loads(content_features['entity_types_json'])
            
            assert len(entities) == 1
            assert entities[0]['text'] == 'Python'
            assert entity_types['PRODUCT'] == 1

    def test_content_analysis_with_analyzer_failure(self, mock_vault_reader, mock_vector_encoder):
        """Test content analysis when analyzer fails."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        # Mock content analyzer to raise exception
        with patch.object(generator.content_analyzer, 'analyze_content', side_effect=Exception("Analysis failed")):
            note_data = NoteData(
                path="test.md",
                title="Test Note",
                content="Test content",
                metadata={},
                tags=[],
                outgoing_links=[],
                word_count=2
            )
            
            # Should handle error gracefully and return default values
            content_features = generator._compute_content_features(note_data)
            
            assert content_features['sentiment_score'] == 0.0
            assert content_features['sentiment_label'] == 'neutral'
            assert content_features['readability_score'] == 0.0
            assert content_features['content_type'] == 'general'


if __name__ == "__main__":
    pytest.main([__file__])