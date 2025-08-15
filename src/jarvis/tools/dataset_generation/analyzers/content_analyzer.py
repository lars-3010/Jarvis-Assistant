"""
Content analyzer for comprehensive text analysis.

This module provides advanced content analysis including sentiment analysis,
readability metrics, named entity recognition, and content complexity scoring.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from jarvis.utils.logging import setup_logging
from ..error_handling import (
    with_error_handling, ComponentType, ErrorSeverity, FallbackValues,
    get_error_tracker
)

logger = setup_logging(__name__)


@dataclass
class ContentFeatures:
    """Comprehensive content analysis features."""
    # Sentiment analysis
    sentiment_score: float = 0.0  # -1 (negative) to 1 (positive)
    sentiment_label: str = "neutral"  # positive, negative, neutral
    
    # Readability metrics
    readability_score: float = 0.0  # Flesch Reading Ease (0-100)
    readability_grade: float = 0.0  # Grade level
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    
    # Complexity indicators
    complexity_score: float = 0.0  # 0-1 normalized complexity
    technical_density: float = 0.0  # Ratio of technical terms
    concept_density: float = 0.0  # Ratio of concept words
    
    # Named entities
    named_entities: List[Dict[str, str]] = field(default_factory=list)
    entity_types: Dict[str, int] = field(default_factory=dict)  # Count by type
    
    # Content structure
    heading_count: int = 0
    max_heading_depth: int = 0
    list_count: int = 0
    code_block_count: int = 0
    
    # Vocabulary analysis
    unique_words: int = 0
    total_words: int = 0
    vocabulary_richness: float = 0.0  # unique/total ratio
    
    # Content type classification
    content_type: str = "general"  # technical, academic, creative, etc.
    domain_indicators: List[str] = field(default_factory=list)


class ContentAnalyzer:
    """Comprehensive content analysis for notes."""
    
    def __init__(self, use_spacy: bool = True, use_textstat: bool = True):
        """Initialize the content analyzer.
        
        Args:
            use_spacy: Whether to use spaCy for NLP features
            use_textstat: Whether to use textstat for readability metrics
        """
        self.use_spacy = use_spacy
        self.use_textstat = use_textstat
        self._nlp = None
        self._spacy_available = False
        self._textstat_available = False
        
        # Initialize spaCy if available
        if use_spacy:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                    self._spacy_available = True
                    logger.info("spaCy English model loaded successfully")
                except OSError:
                    logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    self._spacy_available = False
            except ImportError:
                logger.warning("spaCy not available. Install with: pip install spacy")
                self._spacy_available = False
        
        # Check textstat availability
        if use_textstat:
            try:
                import textstat
                self._textstat_available = True
                logger.info("textstat library available")
            except ImportError:
                logger.warning("textstat not available. Install with: pip install textstat")
                self._textstat_available = False
        
        # Technical term patterns
        self._technical_patterns = [
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b',  # CamelCase
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:_\w+)+\b',  # snake_case
            r'\b\w*[Tt]ech\w*\b',  # Tech-related words
            r'\b\w*[Aa]lgorithm\w*\b',  # Algorithm-related
            r'\b\w*[Dd]ata\w*\b',  # Data-related
        ]
        
        # Domain indicators
        self._domain_patterns = {
            'technical': [
                'algorithm', 'function', 'class', 'method', 'implementation',
                'code', 'programming', 'software', 'system', 'architecture'
            ],
            'academic': [
                'research', 'study', 'analysis', 'theory', 'hypothesis',
                'methodology', 'findings', 'conclusion', 'literature'
            ],
            'business': [
                'strategy', 'market', 'customer', 'revenue', 'profit',
                'business', 'management', 'organization', 'process'
            ],
            'creative': [
                'design', 'creative', 'art', 'visual', 'aesthetic',
                'inspiration', 'concept', 'idea', 'innovation'
            ],
            'scientific': [
                'experiment', 'hypothesis', 'data', 'results', 'method',
                'observation', 'measurement', 'analysis', 'conclusion'
            ]
        }
    
    def analyze_content(self, text: str) -> ContentFeatures:
        """Perform comprehensive content analysis with robust error handling.
        
        Args:
            text: Text content to analyze
            
        Returns:
            ContentFeatures object with all analysis results
        """
        if not text or not text.strip():
            return ContentFeatures()
        
        features = ContentFeatures()
        
        # Basic text preprocessing with error handling
        clean_text = self._clean_text_safe(text)
        
        # Sentiment analysis with error handling
        sentiment_data = self._analyze_sentiment_safe(clean_text)
        features.sentiment_score = sentiment_data[0]
        features.sentiment_label = sentiment_data[1]
        
        # Readability metrics with error handling
        readability_metrics = self._analyze_readability_safe(clean_text)
        features.readability_score = readability_metrics['flesch_score']
        features.readability_grade = readability_metrics['grade_level']
        features.avg_sentence_length = readability_metrics['avg_sentence_length']
        features.avg_word_length = readability_metrics['avg_word_length']
        
        # Named entity recognition with error handling
        if self._spacy_available:
            entities_data = self._extract_named_entities_safe(clean_text)
            features.named_entities = entities_data['entities']
            features.entity_types = entities_data['types']
        
        # Content structure analysis with error handling
        structure_data = self._analyze_structure_safe(text)
        features.heading_count = structure_data['heading_count']
        features.max_heading_depth = structure_data['max_heading_depth']
        features.list_count = structure_data['list_count']
        features.code_block_count = structure_data['code_block_count']
        
        # Vocabulary analysis with error handling
        vocab_data = self._analyze_vocabulary_safe(clean_text)
        features.unique_words = vocab_data['unique_words']
        features.total_words = vocab_data['total_words']
        features.vocabulary_richness = vocab_data['richness']
        
        # Technical and concept density with error handling
        features.technical_density = self._calculate_technical_density_safe(clean_text)
        features.concept_density = self._calculate_concept_density_safe(clean_text)
        
        # Complexity scoring with error handling
        features.complexity_score = self._calculate_complexity_score_safe(features)
        
        # Content type classification with error handling
        content_type_data = self._classify_content_type_safe(clean_text)
        features.content_type = content_type_data[0]
        features.domain_indicators = content_type_data[1]
        
        return features
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="text_cleaning",
        fallback_value="",
        severity=ErrorSeverity.LOW,
        recovery_action="Text will be used as-is without cleaning"
    )
    def _clean_text_safe(self, text: str) -> str:
        """Safely clean text with error handling."""
        return self._clean_text(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="sentiment_analysis",
        fallback_value=(FallbackValues.SENTIMENT_SCORE, FallbackValues.SENTIMENT_LABEL),
        severity=ErrorSeverity.LOW,
        recovery_action="Sentiment will be marked as neutral"
    )
    def _analyze_sentiment_safe(self, text: str) -> Tuple[float, str]:
        """Safely analyze sentiment with error handling."""
        return self._analyze_sentiment(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="readability_analysis",
        fallback_value={
            'flesch_score': FallbackValues.READABILITY_SCORE,
            'grade_level': FallbackValues.READABILITY_GRADE,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0
        },
        severity=ErrorSeverity.LOW,
        recovery_action="Readability metrics will use default values"
    )
    def _analyze_readability_safe(self, text: str) -> Dict[str, float]:
        """Safely analyze readability with error handling."""
        return self._analyze_readability(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="named_entity_extraction",
        fallback_value={'entities': FallbackValues.NAMED_ENTITIES, 'types': FallbackValues.ENTITY_TYPES},
        severity=ErrorSeverity.LOW,
        recovery_action="Named entities will be empty"
    )
    def _extract_named_entities_safe(self, text: str) -> Dict[str, Any]:
        """Safely extract named entities with error handling."""
        return self._extract_named_entities(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="structure_analysis",
        fallback_value={
            'heading_count': FallbackValues.HEADING_COUNT,
            'max_heading_depth': FallbackValues.MAX_HEADING_DEPTH,
            'list_count': 0,
            'code_block_count': 0
        },
        severity=ErrorSeverity.LOW,
        recovery_action="Structure metrics will use default values"
    )
    def _analyze_structure_safe(self, text: str) -> Dict[str, int]:
        """Safely analyze structure with error handling."""
        return self._analyze_structure(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="vocabulary_analysis",
        fallback_value={
            'unique_words': 0,
            'total_words': 0,
            'richness': FallbackValues.VOCABULARY_RICHNESS
        },
        severity=ErrorSeverity.LOW,
        recovery_action="Vocabulary metrics will use default values"
    )
    def _analyze_vocabulary_safe(self, text: str) -> Dict[str, Any]:
        """Safely analyze vocabulary with error handling."""
        return self._analyze_vocabulary(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="technical_density",
        fallback_value=FallbackValues.TECHNICAL_DENSITY,
        severity=ErrorSeverity.LOW,
        recovery_action="Technical density will be set to 0"
    )
    def _calculate_technical_density_safe(self, text: str) -> float:
        """Safely calculate technical density with error handling."""
        return self._calculate_technical_density(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="concept_density",
        fallback_value=FallbackValues.CONCEPT_DENSITY,
        severity=ErrorSeverity.LOW,
        recovery_action="Concept density will be set to 0"
    )
    def _calculate_concept_density_safe(self, text: str) -> float:
        """Safely calculate concept density with error handling."""
        return self._calculate_concept_density(text)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="complexity_scoring",
        fallback_value=FallbackValues.COMPLEXITY_SCORE,
        severity=ErrorSeverity.LOW,
        recovery_action="Complexity score will be set to medium (0.5)"
    )
    def _calculate_complexity_score_safe(self, features: ContentFeatures) -> float:
        """Safely calculate complexity score with error handling."""
        return self._calculate_complexity_score(features)
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="content_type_classification",
        fallback_value=(FallbackValues.CONTENT_TYPE, []),
        severity=ErrorSeverity.LOW,
        recovery_action="Content type will be marked as general"
    )
    def _classify_content_type_safe(self, text: str) -> Tuple[str, List[str]]:
        """Safely classify content type with error handling."""
        return self._classify_content_type(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis while preserving structure."""
        if not text:
            return ""
        
        # Remove markdown formatting but keep structure
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using rule-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not text:
            return 0.0, "neutral"
        
        # Simple rule-based sentiment analysis
        # In a production system, you'd use a proper sentiment analysis model
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
            'success', 'achieve', 'accomplish', 'improve', 'better', 'best'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'sad', 'angry', 'frustrated', 'disappointed', 'fail', 'failure',
            'problem', 'issue', 'difficult', 'hard', 'struggle', 'worst'
        ]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0, "neutral"
        
        sentiment_score = (positive_count - negative_count) / len(words)
        sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))  # Scale and clamp
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return sentiment_score, label
    
    def _analyze_readability(self, text: str) -> Dict[str, float]:
        """Analyze readability metrics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        metrics = {
            'flesch_score': 0.0,
            'grade_level': 0.0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0
        }
        
        if not text:
            return metrics
        
        try:
            # Use textstat if available
            if self._textstat_available:
                import textstat
                metrics['flesch_score'] = textstat.flesch_reading_ease(text)
                metrics['grade_level'] = textstat.flesch_kincaid_grade(text)
            else:
                # Fallback to simple calculations
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                words = text.split()
                
                if sentences and words:
                    metrics['avg_sentence_length'] = len(words) / len(sentences)
                    metrics['avg_word_length'] = sum(len(word) for word in words) / len(words)
                    
                    # Simple Flesch approximation
                    asl = metrics['avg_sentence_length']
                    awl = metrics['avg_word_length']
                    metrics['flesch_score'] = max(0, 206.835 - (1.015 * asl) - (84.6 * (awl / 4.7)))
                    metrics['grade_level'] = max(0, (0.39 * asl) + (11.8 * (awl / 4.7)) - 15.59)
        
        except Exception as e:
            logger.warning(f"Error calculating readability metrics: {e}")
        
        return metrics
    
    def _extract_named_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities using spaCy.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entities and type counts
        """
        entities_data = {
            'entities': [],
            'types': {}
        }
        
        if not self._spacy_available or not text:
            return entities_data
        
        try:
            import spacy
            doc = self._nlp(text)
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_) or ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                entities_data['entities'].append(entity_info)
                
                # Count by type
                label = ent.label_
                entities_data['types'][label] = entities_data['types'].get(label, 0) + 1
        
        except Exception as e:
            logger.warning(f"Error extracting named entities: {e}")
        
        return entities_data
    
    def _analyze_structure(self, text: str) -> Dict[str, int]:
        """Analyze document structure.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with structure metrics
        """
        structure = {
            'heading_count': 0,
            'max_heading_depth': 0,
            'list_count': 0,
            'code_block_count': 0
        }
        
        if not text:
            return structure
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Count headings
            if line.startswith('#'):
                structure['heading_count'] += 1
                depth = len(line) - len(line.lstrip('#'))
                structure['max_heading_depth'] = max(structure['max_heading_depth'], depth)
            
            # Count list items
            elif line.startswith(('-', '*', '+')):
                structure['list_count'] += 1
            elif re.match(r'^\d+\.', line):
                structure['list_count'] += 1
            
            # Count code blocks
            elif line.startswith('```'):
                structure['code_block_count'] += 1
        
        # Code blocks come in pairs
        structure['code_block_count'] = structure['code_block_count'] // 2
        
        return structure
    
    def _analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """Analyze vocabulary richness and complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with vocabulary metrics
        """
        vocab_data = {
            'unique_words': 0,
            'total_words': 0,
            'richness': 0.0
        }
        
        if not text:
            return vocab_data
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        vocab_data['total_words'] = len(words)
        vocab_data['unique_words'] = len(set(words))
        
        if vocab_data['total_words'] > 0:
            vocab_data['richness'] = vocab_data['unique_words'] / vocab_data['total_words']
        
        return vocab_data
    
    def _calculate_technical_density(self, text: str) -> float:
        """Calculate density of technical terms.
        
        Args:
            text: Text to analyze
            
        Returns:
            Technical density ratio (0-1)
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        technical_count = 0
        for pattern in self._technical_patterns:
            technical_count += len(re.findall(pattern, text))
        
        return min(1.0, technical_count / len(words))
    
    def _calculate_concept_density(self, text: str) -> float:
        """Calculate density of conceptual terms.
        
        Args:
            text: Text to analyze
            
        Returns:
            Concept density ratio (0-1)
        """
        if not text:
            return 0.0
        
        # Concept indicators (capitalized words, domain-specific terms)
        concept_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b(?:concept|theory|principle|framework|model|approach)\b',
            r'\b(?:analysis|synthesis|evaluation|application)\b'
        ]
        
        words = text.split()
        if not words:
            return 0.0
        
        concept_count = 0
        for pattern in concept_patterns:
            concept_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return min(1.0, concept_count / len(words))
    
    def _calculate_complexity_score(self, features: ContentFeatures) -> float:
        """Calculate overall complexity score.
        
        Args:
            features: ContentFeatures object with metrics
            
        Returns:
            Normalized complexity score (0-1)
        """
        # Combine multiple factors into complexity score
        factors = []
        
        # Readability (inverse - lower readability = higher complexity)
        if features.readability_score > 0:
            readability_complexity = max(0, (100 - features.readability_score) / 100)
            factors.append(readability_complexity)
        
        # Technical density
        factors.append(features.technical_density)
        
        # Concept density
        factors.append(features.concept_density)
        
        # Vocabulary richness
        factors.append(features.vocabulary_richness)
        
        # Average sentence length (normalized)
        if features.avg_sentence_length > 0:
            sentence_complexity = min(1.0, features.avg_sentence_length / 25.0)
            factors.append(sentence_complexity)
        
        # Entity density (more entities = more complex)
        if features.total_words > 0:
            entity_density = min(1.0, len(features.named_entities) / features.total_words * 10)
            factors.append(entity_density)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _classify_content_type(self, text: str) -> Tuple[str, List[str]]:
        """Classify content type based on domain indicators.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (content_type, domain_indicators)
        """
        if not text:
            return "general", []
        
        text_lower = text.lower()
        domain_scores = {}
        found_indicators = []
        
        # Score each domain
        for domain, indicators in self._domain_patterns.items():
            score = 0
            domain_indicators = []
            
            for indicator in indicators:
                count = len(re.findall(r'\b' + re.escape(indicator) + r'\b', text_lower))
                if count > 0:
                    score += count
                    domain_indicators.append(indicator)
            
            if score > 0:
                domain_scores[domain] = score
                found_indicators.extend(domain_indicators)
        
        # Determine primary content type
        if domain_scores:
            content_type = max(domain_scores, key=domain_scores.get)
        else:
            content_type = "general"
        
        return content_type, list(set(found_indicators))
    
    def get_analysis_summary(self, features: ContentFeatures) -> Dict[str, Any]:
        """Get a summary of the content analysis.
        
        Args:
            features: ContentFeatures object
            
        Returns:
            Dictionary with analysis summary
        """
        return {
            'content_type': features.content_type,
            'complexity_level': 'high' if features.complexity_score > 0.7 else 'medium' if features.complexity_score > 0.4 else 'low',
            'readability_level': 'easy' if features.readability_score > 70 else 'moderate' if features.readability_score > 50 else 'difficult',
            'sentiment': features.sentiment_label,
            'technical_content': features.technical_density > 0.1,
            'entity_count': len(features.named_entities),
            'structure_richness': 'high' if features.heading_count > 5 else 'medium' if features.heading_count > 2 else 'low',
            'vocabulary_richness': 'high' if features.vocabulary_richness > 0.7 else 'medium' if features.vocabulary_richness > 0.5 else 'low'
        }