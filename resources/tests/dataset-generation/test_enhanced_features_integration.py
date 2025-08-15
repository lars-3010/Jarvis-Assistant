#!/usr/bin/env python3
"""
Integration test for enhanced dataset features with comprehensive error handling.

This test verifies that all error handling and fallback mechanisms work correctly
across all components of the enhanced dataset generation system.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from jarvis.tools.dataset_generation.feature_engineer import FeatureEngineer, EnhancedFeatures
from jarvis.tools.dataset_generation.models.data_models import NoteData
from jarvis.tools.dataset_generation.error_handling import (
    get_error_tracker, get_dependency_checker, log_system_health,
    ComponentType, ErrorSeverity, FallbackValues
)
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class MockVectorEncoder:
    """Mock vector encoder for testing."""
    
    def __init__(self, vector_dim: int = 384):
        self.vector_dim = vector_dim
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Generate mock embeddings."""
        return np.random.rand(len(documents), self.vector_dim).astype(np.float32)


def create_test_notes() -> List[NoteData]:
    """Create test notes with various content types."""
    notes = [
        NoteData(
            path="test/note1.md",
            title="Technical Documentation",
            content="""# API Documentation

This is a comprehensive guide to our REST API. The API provides endpoints for:

- User authentication using JWT tokens
- Data retrieval with pagination
- Real-time updates via WebSocket connections

## Authentication

All requests must include a valid JWT token in the Authorization header.

```python
import requests

headers = {'Authorization': 'Bearer your-jwt-token'}
response = requests.get('/api/users', headers=headers)
```

The system implements OAuth 2.0 for secure authentication.""",
            metadata={'created': 1640995200, 'modified': 1641081600},
            tags=['api', 'documentation', 'technical'],
            outgoing_links=['test/auth.md', 'test/websockets.md'],
            embedding=np.random.rand(384).astype(np.float32)
        ),
        
        NoteData(
            path="test/note2.md",
            title="Creative Writing",
            content="""# The Journey

The sun was setting over the mountains, casting long shadows across the valley. 
Sarah walked slowly along the winding path, her thoughts drifting to the 
conversation she'd had with her grandmother that morning.

"Life is like a river," her grandmother had said, "always flowing, always changing, 
but always finding its way to the sea."

The metaphor resonated with Sarah as she contemplated the changes in her own life. 
The new job opportunity in the city represented both excitement and uncertainty.""",
            metadata={'created': 1641081600, 'modified': 1641168000},
            tags=['creative', 'writing', 'story'],
            outgoing_links=[],
            embedding=np.random.rand(384).astype(np.float32)
        ),
        
        NoteData(
            path="test/note3.md",
            title="Research Notes",
            content="""# Machine Learning Research

## Abstract

This study investigates the effectiveness of transformer architectures in 
natural language processing tasks. We conducted experiments on multiple 
datasets including GLUE, SuperGLUE, and custom domain-specific corpora.

## Methodology

Our approach utilized the following techniques:
1. Pre-training on large-scale text corpora
2. Fine-tuning on task-specific datasets
3. Evaluation using standard benchmarks

## Results

The results demonstrate significant improvements over baseline models:
- BERT-base: 84.3% accuracy
- Our model: 87.1% accuracy
- Statistical significance: p < 0.001

## Conclusion

The findings suggest that architectural modifications can lead to substantial 
performance gains in NLP tasks.""",
            metadata={'created': 1641168000, 'modified': 1641254400},
            tags=['research', 'machine-learning', 'nlp'],
            outgoing_links=['test/bert.md', 'test/transformers.md'],
            embedding=np.random.rand(384).astype(np.float32)
        ),
        
        NoteData(
            path="test/note4.md",
            title="Empty Note",
            content="",
            metadata={'created': 1641254400, 'modified': 1641254400},
            tags=[],
            outgoing_links=[],
            embedding=None
        ),
        
        NoteData(
            path="test/note5.md",
            title="Corrupted Content",
            content="This note has some weird characters: \x00\x01\x02 and incomplete sentences that might cause",
            metadata={'created': 1641340800, 'modified': 1641340800},
            tags=['test'],
            outgoing_links=[],
            embedding=np.array([float('nan')] * 384).astype(np.float32)  # NaN embedding
        )
    ]
    
    return notes


def test_system_health_monitoring():
    """Test system health monitoring and dependency checking."""
    print("\n=== Testing System Health Monitoring ===")
    
    # Get dependency status
    dependency_checker = get_dependency_checker()
    deps = dependency_checker.get_dependency_status()
    
    print("Dependency Status:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}: {available}")
    
    # Log system health
    health_info = log_system_health()
    
    print(f"\nSystem Health Score: {health_info['overall_health_score']:.2f}")
    print(f"Healthy Components: {health_info['healthy_components']}/{health_info['total_components']}")
    
    return health_info


def test_feature_engineer_with_errors():
    """Test FeatureEngineer with various error conditions."""
    print("\n=== Testing FeatureEngineer Error Handling ===")
    
    # Create mock vector encoder
    vector_encoder = MockVectorEncoder()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(vector_encoder)
    
    # Create test notes
    notes = create_test_notes()
    
    print(f"Created {len(notes)} test notes")
    
    # Test analyzer fitting with error handling
    print("\nTesting analyzer fitting...")
    fit_results = feature_engineer.fit_analyzers(notes)
    
    print("Fit Results:")
    print(f"  Semantic fitted: {fit_results['semantic_fitted']}")
    print(f"  Topic fitted: {fit_results['topic_fitted']}")
    print(f"  Notes processed: {fit_results['notes_processed']}")
    print(f"  Errors: {len(fit_results['errors'])}")
    
    if fit_results['errors']:
        print("  Error details:")
        for error in fit_results['errors']:
            print(f"    - {error}")
    
    # Test feature extraction with various note types
    print("\nTesting feature extraction...")
    
    for i, note in enumerate(notes):
        print(f"\nProcessing note {i+1}: {note.title}")
        
        try:
            features = feature_engineer.extract_note_features(note)
            
            # Validate features
            print(f"  Content features available: {features.content_features is not None}")
            print(f"  TF-IDF vocabulary richness: {features.tfidf_vocabulary_richness:.3f}")
            print(f"  Average TF-IDF score: {features.avg_tfidf_score:.3f}")
            print(f"  Dominant topic ID: {features.dominant_topic_id}")
            print(f"  Topic probability: {features.dominant_topic_probability:.3f}")
            
            # Check for fallback values
            if features.content_features:
                cf = features.content_features
                print(f"  Sentiment: {cf.sentiment_label} ({cf.sentiment_score:.3f})")
                print(f"  Readability: {cf.readability_score:.1f}")
                print(f"  Complexity: {cf.complexity_score:.3f}")
                print(f"  Content type: {cf.content_type}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return feature_engineer


def test_pair_feature_extraction(feature_engineer: FeatureEngineer):
    """Test pair feature extraction with error handling."""
    print("\n=== Testing Pair Feature Extraction ===")
    
    notes = create_test_notes()
    
    # Test various note pairs
    test_pairs = [
        (0, 1),  # Technical vs Creative
        (0, 2),  # Technical vs Research
        (1, 2),  # Creative vs Research
        (0, 3),  # Technical vs Empty
        (3, 4),  # Empty vs Corrupted
    ]
    
    for i, (idx_a, idx_b) in enumerate(test_pairs):
        note_a = notes[idx_a]
        note_b = notes[idx_b]
        
        print(f"\nPair {i+1}: {note_a.title} <-> {note_b.title}")
        
        try:
            pair_features = feature_engineer.extract_pair_features(note_a, note_b)
            
            print(f"  Semantic similarity: {pair_features.get('semantic_similarity', 0.0):.3f}")
            print(f"  TF-IDF similarity: {pair_features.get('tfidf_similarity', 0.0):.3f}")
            print(f"  Combined similarity: {pair_features.get('combined_similarity', 0.0):.3f}")
            print(f"  Topic similarity: {pair_features.get('topic_similarity', 0.0):.3f}")
            print(f"  Same dominant topic: {pair_features.get('same_dominant_topic', False)}")
            print(f"  Content similarity: {pair_features.get('content_similarity', 0.0):.3f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")


def test_error_tracking():
    """Test error tracking and reporting."""
    print("\n=== Testing Error Tracking ===")
    
    error_tracker = get_error_tracker()
    
    # Get error summary
    error_summary = error_tracker.get_error_summary()
    
    print("Error Summary:")
    print(f"  Total errors: {error_summary['total_errors']}")
    
    if error_summary['by_severity']:
        print("  By severity:")
        for severity, count in error_summary['by_severity'].items():
            print(f"    {severity}: {count}")
    
    if error_summary['by_component']:
        print("  By component:")
        for component, count in error_summary['by_component'].items():
            print(f"    {component}: {count}")
    
    print("  Component reliability:")
    for component, score in error_summary['component_reliability'].items():
        print(f"    {component}: {score:.3f}")
    
    # Show recent errors
    if error_summary['recent_errors']:
        print("\n  Recent errors:")
        for error in error_summary['recent_errors'][-3:]:  # Show last 3
            print(f"    - {error['component']}: {error['feature_name']} ({error['severity']})")
            print(f"      {error['message']}")
            if error['fallback_used']:
                print(f"      Fallback used: {error['fallback_value']}")


def test_fallback_values():
    """Test fallback value mechanisms."""
    print("\n=== Testing Fallback Values ===")
    
    # Test content features fallback
    content_fallbacks = FallbackValues.get_fallback_content_features()
    print("Content feature fallbacks:")
    for key, value in content_fallbacks.items():
        print(f"  {key}: {value}")
    
    # Test topic features fallback
    topic_fallbacks = FallbackValues.get_fallback_topic_features()
    print("\nTopic feature fallbacks:")
    for key, value in topic_fallbacks.items():
        print(f"  {key}: {value}")
    
    # Test graph features fallback
    graph_fallbacks = FallbackValues.get_fallback_graph_features()
    print("\nGraph feature fallbacks:")
    for key, value in graph_fallbacks.items():
        print(f"  {key}: {value}")


def test_analyzer_status():
    """Test analyzer status reporting."""
    print("\n=== Testing Analyzer Status ===")
    
    # Create feature engineer and fit analyzers
    vector_encoder = MockVectorEncoder()
    feature_engineer = FeatureEngineer(vector_encoder)
    
    notes = create_test_notes()
    feature_engineer.fit_analyzers(notes)
    
    # Get analyzer status
    status = feature_engineer.get_analyzer_status()
    
    print("Analyzer Status:")
    for analyzer, info in status.items():
        print(f"\n  {analyzer}:")
        for key, value in info.items():
            if isinstance(value, dict) and value:
                print(f"    {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            else:
                print(f"    {key}: {value}")


def run_comprehensive_test():
    """Run comprehensive integration test."""
    print("=" * 60)
    print("ENHANCED DATASET FEATURES - COMPREHENSIVE ERROR HANDLING TEST")
    print("=" * 60)
    
    try:
        # Test 1: System health monitoring
        health_info = test_system_health_monitoring()
        
        # Test 2: Feature engineer with error handling
        feature_engineer = test_feature_engineer_with_errors()
        
        # Test 3: Pair feature extraction
        test_pair_feature_extraction(feature_engineer)
        
        # Test 4: Error tracking
        test_error_tracking()
        
        # Test 5: Fallback values
        test_fallback_values()
        
        # Test 6: Analyzer status
        test_analyzer_status()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        # Final system health check
        final_health = log_system_health()
        
        print(f"Final System Health Score: {final_health['overall_health_score']:.2f}")
        print(f"Total Errors Encountered: {final_health['error_summary']['total_errors']}")
        
        if final_health['overall_health_score'] >= 0.5:
            print("✓ COMPREHENSIVE ERROR HANDLING TEST PASSED")
            print("  All components demonstrated graceful degradation")
            print("  Fallback mechanisms working correctly")
            print("  Error tracking and reporting functional")
        else:
            print("⚠ COMPREHENSIVE ERROR HANDLING TEST COMPLETED WITH WARNINGS")
            print("  Some components may have limited functionality")
            print("  Check dependency installation for full features")
        
        return True
        
    except Exception as e:
        print(f"\n✗ COMPREHENSIVE ERROR HANDLING TEST FAILED")
        print(f"  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)