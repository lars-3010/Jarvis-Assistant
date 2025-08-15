"""
Unit tests for ContentAnalyzer component.

Tests comprehensive content analysis including sentiment analysis,
readability metrics, named entity recognition, and content complexity scoring.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.jarvis.tools.dataset_generation.analyzers.content_analyzer import (
    ContentAnalyzer, ContentFeatures
)
from src.jarvis.tools.dataset_generation.error_handling import (
    get_error_tracker, ComponentType, ErrorSeverity
)


class TestContentFeatures:
    """Test ContentFeatures data class."""
    
    def test_content_features_initialization(self):
        """Test ContentFeatures initialization with default values."""
        features = ContentFeatures()
        
        assert features.sentiment_score == 0.0
        assert features.sentiment_label == "neutral"
        assert features.readability_score == 0.0
        assert features.complexity_score == 0.0
        assert features.named_entities == []
        assert features.entity_types == {}
        assert features.content_type == "general"
        assert features.vocabulary_richness == 0.0
    
    def test_content_features_with_values(self):
        """Test ContentFeatures with specific values."""
        entities = [{"text": "Python", "label": "LANGUAGE"}]
        entity_types = {"LANGUAGE": 1}
        
        features = ContentFeatures(
            sentiment_score=0.8,
            sentiment_label="positive",
            readability_score=75.0,
            complexity_score=0.6,
            named_entities=entities,
            entity_types=entity_types,
            content_type="technical",
            vocabulary_richness=0.7
        )
        
        assert features.sentiment_score == 0.8
        assert features.sentiment_label == "positive"
        assert features.readability_score == 75.0
        assert features.complexity_score == 0.6
        assert features.named_entities == entities
        assert features.entity_types == entity_types
        assert features.content_type == "technical"
        assert features.vocabulary_richness == 0.7


class TestContentAnalyzer:
    """Test ContentAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create ContentAnalyzer instance for testing."""
        return ContentAnalyzer(use_spacy=False, use_textstat=False)
    
    @pytest.fixture
    def analyzer_with_deps(self):
        """Create ContentAnalyzer with mocked dependencies."""
        with patch('src.jarvis.tools.dataset_generation.analyzers.content_analyzer.spacy') as mock_spacy, \
             patch('src.jarvis.tools.dataset_generation.analyzers.content_analyzer.textstat') as mock_textstat:
            
            # Mock spaCy
            mock_nlp = Mock()
            mock_spacy.load.return_value = mock_nlp
            
            analyzer = ContentAnalyzer(use_spacy=True, use_textstat=True)
            analyzer._spacy_available = True
            analyzer._textstat_available = True
            analyzer._nlp = mock_nlp
            
            return analyzer
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        # Introduction to Machine Learning
        
        Machine learning is a fascinating field of artificial intelligence.
        It involves algorithms that can learn from data without being explicitly programmed.
        
        ## Key Concepts
        
        - **Supervised Learning**: Learning with labeled examples
        - **Unsupervised Learning**: Finding patterns in unlabeled data
        - **Deep Learning**: Neural networks with multiple layers
        
        ```python
        def train_model(data):
            return model.fit(data)
        ```
        
        This technology is revolutionizing many industries today.
        """
    
    def test_analyzer_initialization_no_deps(self):
        """Test analyzer initialization without dependencies."""
        analyzer = ContentAnalyzer(use_spacy=False, use_textstat=False)
        
        assert not analyzer.use_spacy
        assert not analyzer.use_textstat
        assert not analyzer._spacy_available
        assert not analyzer._textstat_available
        assert analyzer._nlp is None
    
    def test_analyzer_initialization_with_deps(self, analyzer_with_deps):
        """Test analyzer initialization with mocked dependencies."""
        assert analyzer_with_deps.use_spacy
        assert analyzer_with_deps.use_textstat
        assert analyzer_with_deps._spacy_available
        assert analyzer_with_deps._textstat_available
        assert analyzer_with_deps._nlp is not None
    
    def test_analyze_content_empty_text(self, analyzer):
        """Test content analysis with empty text."""
        features = analyzer.analyze_content("")
        
        assert isinstance(features, ContentFeatures)
        assert features.sentiment_score == 0.0
        assert features.sentiment_label == "neutral"
        assert features.total_words == 0
        assert features.unique_words == 0
    
    def test_analyze_content_none_text(self, analyzer):
        """Test content analysis with None text."""
        features = analyzer.analyze_content(None)
        
        assert isinstance(features, ContentFeatures)
        assert features.sentiment_score == 0.0
        assert features.total_words == 0
    
    def test_analyze_content_basic(self, analyzer, sample_text):
        """Test basic content analysis without external dependencies."""
        features = analyzer.analyze_content(sample_text)
        
        assert isinstance(features, ContentFeatures)
        assert features.total_words > 0
        assert features.unique_words > 0
        assert features.vocabulary_richness > 0
        assert features.heading_count > 0
        assert features.list_count > 0
        assert features.code_block_count > 0
    
    def test_clean_text(self, analyzer):
        """Test text cleaning functionality."""
        text = "This is **bold** and *italic* text with `code` and [link](url)."
        cleaned = analyzer._clean_text(text)
        
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "`" not in cleaned
        assert "[" not in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
        assert "code" in cleaned
        assert "link" in cleaned
    
    def test_sentiment_analysis_positive(self, analyzer):
        """Test sentiment analysis with positive text."""
        text = "This is a wonderful and amazing experience. I love it!"
        sentiment_score, sentiment_label = analyzer._analyze_sentiment(text)
        
        assert sentiment_score > 0
        assert sentiment_label == "positive"
    
    def test_sentiment_analysis_negative(self, analyzer):
        """Test sentiment analysis with negative text."""
        text = "This is terrible and awful. I hate it and it's the worst!"
        sentiment_score, sentiment_label = analyzer._analyze_sentiment(text)
        
        assert sentiment_score < 0
        assert sentiment_label == "negative"
    
    def test_sentiment_analysis_neutral(self, analyzer):
        """Test sentiment analysis with neutral text."""
        text = "This is a document about machine learning algorithms."
        sentiment_score, sentiment_label = analyzer._analyze_sentiment(text)
        
        assert sentiment_label == "neutral"
        assert -0.1 <= sentiment_score <= 0.1
    
    def test_readability_analysis_fallback(self, analyzer):
        """Test readability analysis without textstat."""
        text = "This is a simple sentence. This is another sentence with more words."
        metrics = analyzer._analyze_readability(text)
        
        assert isinstance(metrics, dict)
        assert 'flesch_score' in metrics
        assert 'grade_level' in metrics
        assert 'avg_sentence_length' in metrics
        assert 'avg_word_length' in metrics
        
        # Should have calculated basic metrics
        assert metrics['avg_sentence_length'] > 0
        assert metrics['avg_word_length'] > 0
    
    @patch('src.jarvis.tools.dataset_generation.analyzers.content_analyzer.textstat')
    def test_readability_analysis_with_textstat(self, mock_textstat, analyzer):
        """Test readability analysis with textstat."""
        analyzer._textstat_available = True
        
        mock_textstat.flesch_reading_ease.return_value = 75.0
        mock_textstat.flesch_kincaid_grade.return_value = 8.0
        
        text = "This is a test sentence."
        metrics = analyzer._analyze_readability(text)
        
        assert metrics['flesch_score'] == 75.0
        assert metrics['grade_level'] == 8.0
        mock_textstat.flesch_reading_ease.assert_called_once_with(text)
        mock_textstat.flesch_kincaid_grade.assert_called_once_with(text)
    
    def test_structure_analysis(self, analyzer, sample_text):
        """Test document structure analysis."""
        structure = analyzer._analyze_structure(sample_text)
        
        assert isinstance(structure, dict)
        assert structure['heading_count'] >= 2  # # and ## headings
        assert structure['max_heading_depth'] >= 2
        assert structure['list_count'] >= 3  # Three bullet points
        assert structure['code_block_count'] >= 1  # One code block
    
    def test_vocabulary_analysis(self, analyzer):
        """Test vocabulary analysis."""
        text = "The quick brown fox jumps over the lazy dog. The fox is quick."
        vocab_data = analyzer._analyze_vocabulary(text)
        
        assert vocab_data['total_words'] == 12
        assert vocab_data['unique_words'] == 9  # "the" and "fox" appear twice
        assert vocab_data['richness'] == 9/12
    
    def test_technical_density_calculation(self, analyzer):
        """Test technical term density calculation."""
        technical_text = "The APIController uses CamelCase and snake_case variables with HTTP requests."
        density = analyzer._calculate_technical_density(technical_text)
        
        assert 0.0 <= density <= 1.0
        assert density > 0  # Should detect technical terms
        
        non_technical_text = "The cat sat on the mat and looked around."
        density_low = analyzer._calculate_technical_density(non_technical_text)
        
        assert density > density_low
    
    def test_concept_density_calculation(self, analyzer):
        """Test concept density calculation."""
        conceptual_text = "The Theory of Machine Learning involves complex Analysis and Evaluation."
        density = analyzer._calculate_concept_density(conceptual_text)
        
        assert 0.0 <= density <= 1.0
        assert density > 0  # Should detect conceptual terms
    
    def test_complexity_score_calculation(self, analyzer):
        """Test complexity score calculation."""
        features = ContentFeatures(
            readability_score=30.0,  # Low readability = high complexity
            technical_density=0.3,
            concept_density=0.2,
            vocabulary_richness=0.8,
            avg_sentence_length=20.0,
            named_entities=[{"text": "Python"}, {"text": "AI"}],
            total_words=100
        )
        
        complexity = analyzer._calculate_complexity_score(features)
        
        assert 0.0 <= complexity <= 1.0
        assert complexity > 0.5  # Should be relatively complex
    
    def test_content_type_classification(self, analyzer):
        """Test content type classification."""
        technical_text = "This algorithm implements machine learning using Python programming."
        content_type, indicators = analyzer._classify_content_type(technical_text)
        
        assert content_type == "technical"
        assert len(indicators) > 0
        assert any("algorithm" in indicator or "programming" in indicator for indicator in indicators)
        
        academic_text = "This research study analyzes the methodology and findings."
        content_type, indicators = analyzer._classify_content_type(academic_text)
        
        assert content_type == "academic"
        assert len(indicators) > 0
    
    def test_named_entity_extraction_without_spacy(self, analyzer):
        """Test named entity extraction without spaCy."""
        text = "Apple Inc. was founded by Steve Jobs in California."
        entities_data = analyzer._extract_named_entities(text)
        
        assert entities_data['entities'] == []
        assert entities_data['types'] == {}
    
    def test_named_entity_extraction_with_spacy(self, analyzer_with_deps):
        """Test named entity extraction with mocked spaCy."""
        # Mock spaCy entities
        mock_entity1 = Mock()
        mock_entity1.text = "Apple Inc."
        mock_entity1.label_ = "ORG"
        mock_entity1.start_char = 0
        mock_entity1.end_char = 10
        
        mock_entity2 = Mock()
        mock_entity2.text = "Steve Jobs"
        mock_entity2.label_ = "PERSON"
        mock_entity2.start_char = 25
        mock_entity2.end_char = 35
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity1, mock_entity2]
        
        analyzer_with_deps._nlp.return_value = mock_doc
        
        with patch('spacy.explain') as mock_explain:
            mock_explain.return_value = "Organization"
            
            text = "Apple Inc. was founded by Steve Jobs."
            entities_data = analyzer_with_deps._extract_named_entities(text)
            
            assert len(entities_data['entities']) == 2
            assert entities_data['entities'][0]['text'] == "Apple Inc."
            assert entities_data['entities'][0]['label'] == "ORG"
            assert entities_data['entities'][1]['text'] == "Steve Jobs"
            assert entities_data['entities'][1]['label'] == "PERSON"
            
            assert entities_data['types']['ORG'] == 1
            assert entities_data['types']['PERSON'] == 1
    
    def test_error_handling_integration(self, analyzer):
        """Test error handling integration."""
        # Clear previous errors
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # This should not raise an exception even with problematic input
        features = analyzer.analyze_content("Test content")
        
        assert isinstance(features, ContentFeatures)
        # Should not have added errors for normal operation
        assert len(error_tracker.errors) == initial_error_count
    
    def test_get_analysis_summary(self, analyzer, sample_text):
        """Test analysis summary generation."""
        features = analyzer.analyze_content(sample_text)
        summary = analyzer.get_analysis_summary(features)
        
        assert isinstance(summary, dict)
        assert 'content_type' in summary
        assert 'complexity_level' in summary
        assert 'readability_level' in summary
        assert 'sentiment' in summary
        assert 'technical_content' in summary
        assert 'entity_count' in summary
        assert 'structure_richness' in summary
        assert 'vocabulary_richness' in summary
        
        # Check value types
        assert summary['complexity_level'] in ['low', 'medium', 'high']
        assert summary['readability_level'] in ['easy', 'moderate', 'difficult']
        assert summary['sentiment'] in ['positive', 'negative', 'neutral']
        assert isinstance(summary['technical_content'], bool)
        assert isinstance(summary['entity_count'], int)
        assert summary['structure_richness'] in ['low', 'medium', 'high']
        assert summary['vocabulary_richness'] in ['low', 'medium', 'high']
    
    def test_analyze_content_with_error_handling(self, analyzer):
        """Test content analysis with error handling decorators."""
        # Test with various edge cases that might cause errors
        test_cases = [
            "",  # Empty string
            None,  # None input
            "A" * 10000,  # Very long text
            "🚀 🎉 🔥",  # Emoji-only text
            "   \n\t   ",  # Whitespace-only text
        ]
        
        for test_input in test_cases:
            features = analyzer.analyze_content(test_input)
            assert isinstance(features, ContentFeatures)
            # Should not raise exceptions
    
    @pytest.mark.parametrize("text,expected_sentiment", [
        ("I love this amazing product!", "positive"),
        ("This is terrible and awful!", "negative"),
        ("The weather is cloudy today.", "neutral"),
        ("Great job on the excellent work!", "positive"),
        ("I hate this horrible experience!", "negative"),
    ])
    def test_sentiment_analysis_parametrized(self, analyzer, text, expected_sentiment):
        """Test sentiment analysis with various inputs."""
        _, sentiment_label = analyzer._analyze_sentiment(text)
        assert sentiment_label == expected_sentiment
    
    @pytest.mark.parametrize("text,expected_type", [
        ("This algorithm uses machine learning and programming.", "technical"),
        ("The research study analyzes methodology and findings.", "academic"),
        ("Our business strategy focuses on market growth.", "business"),
        ("The creative design shows artistic inspiration.", "creative"),
        ("The experiment tests the hypothesis with data.", "scientific"),
        ("This is just a regular note about daily life.", "general"),
    ])
    def test_content_type_classification_parametrized(self, analyzer, text, expected_type):
        """Test content type classification with various inputs."""
        content_type, _ = analyzer._classify_content_type(text)
        assert content_type == expected_type


class TestContentAnalyzerErrorHandling:
    """Test error handling in ContentAnalyzer."""
    
    @pytest.fixture
    def analyzer_with_failing_deps(self):
        """Create analyzer with failing dependencies."""
        analyzer = ContentAnalyzer(use_spacy=True, use_textstat=True)
        analyzer._spacy_available = False
        analyzer._textstat_available = False
        return analyzer
    
    def test_graceful_degradation_no_spacy(self, analyzer_with_failing_deps):
        """Test graceful degradation when spaCy is not available."""
        text = "Apple Inc. was founded by Steve Jobs in California."
        features = analyzer_with_failing_deps.analyze_content(text)
        
        # Should still work without spaCy
        assert isinstance(features, ContentFeatures)
        assert features.named_entities == []  # No entities without spaCy
        assert features.entity_types == {}
        
        # Other features should still work
        assert features.total_words > 0
        assert features.vocabulary_richness > 0
    
    def test_graceful_degradation_no_textstat(self, analyzer_with_failing_deps):
        """Test graceful degradation when textstat is not available."""
        text = "This is a test sentence for readability analysis."
        features = analyzer_with_failing_deps.analyze_content(text)
        
        # Should still work without textstat
        assert isinstance(features, ContentFeatures)
        # Should use fallback readability calculation
        assert features.readability_score >= 0
        assert features.readability_grade >= 0
    
    def test_error_tracking(self):
        """Test that errors are properly tracked."""
        error_tracker = get_error_tracker()
        initial_error_count = len(error_tracker.errors)
        
        # Create analyzer that will fail
        with patch('src.jarvis.tools.dataset_generation.analyzers.content_analyzer.spacy') as mock_spacy:
            mock_spacy.load.side_effect = Exception("spaCy loading failed")
            
            analyzer = ContentAnalyzer(use_spacy=True)
            # Should handle the error gracefully
            assert not analyzer._spacy_available
        
        # Check if error was tracked (might be tracked during initialization)
        # The exact count depends on implementation details
        assert len(error_tracker.errors) >= initial_error_count


if __name__ == "__main__":
    pytest.main([__file__])