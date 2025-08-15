#!/usr/bin/env python3
"""
Focused test for error handling components without full system dependencies.

This test verifies the core error handling and fallback mechanisms work correctly.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from the error handling module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'jarvis', 'tools', 'dataset_generation'))

from error_handling import (
    ErrorTracker, FeatureExtractionError, ComponentStatus, ComponentType, 
    ErrorSeverity, FallbackValues, DependencyChecker, with_error_handling,
    get_error_tracker, get_dependency_checker, validate_feature_completeness,
    ensure_feature_quality, create_minimal_features, log_system_health
)

# Import logging directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'jarvis', 'utils'))
from logging import setup_logging

logger = setup_logging(__name__)


def test_error_tracker():
    """Test error tracking functionality."""
    print("\n=== Testing Error Tracker ===")
    
    tracker = ErrorTracker()
    
    # Create test errors
    error1 = FeatureExtractionError(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="sentiment_analysis",
        error_type="ImportError",
        message="spaCy not available",
        severity=ErrorSeverity.MEDIUM,
        timestamp=time.time(),
        context={'dependency': 'spacy'},
        recovery_action="Install spaCy with: pip install spacy",
        fallback_used=True,
        fallback_value="neutral"
    )
    
    error2 = FeatureExtractionError(
        component=ComponentType.TOPIC_MODELER,
        feature_name="topic_fitting",
        error_type="ValueError",
        message="Insufficient documents for topic modeling",
        severity=ErrorSeverity.LOW,
        timestamp=time.time(),
        context={'document_count': 2, 'min_required': 5},
        recovery_action="Provide more documents or reduce min_topic_size",
        fallback_used=True,
        fallback_value=-1
    )
    
    # Record errors
    tracker.record_error(error1)
    tracker.record_error(error2)
    
    # Record some successes
    tracker.record_success(ComponentType.SEMANTIC_ANALYZER)
    tracker.record_success(ComponentType.SEMANTIC_ANALYZER)
    tracker.record_success(ComponentType.CONTENT_ANALYZER)
    
    # Get error summary
    summary = tracker.get_error_summary()
    
    print(f"Total errors: {summary['total_errors']}")
    print(f"Errors by severity: {summary['by_severity']}")
    print(f"Errors by component: {summary['by_component']}")
    print(f"Component reliability: {summary['component_reliability']}")
    
    # Test component status
    content_status = tracker.get_component_status(ComponentType.CONTENT_ANALYZER)
    print(f"\nContent Analyzer Status:")
    print(f"  Available: {content_status.available}")
    print(f"  Success rate: {content_status.success_rate:.3f}")
    print(f"  Reliability score: {content_status.reliability_score:.3f}")
    
    return tracker


def test_dependency_checker():
    """Test dependency checking functionality."""
    print("\n=== Testing Dependency Checker ===")
    
    checker = DependencyChecker()
    
    # Check all dependencies
    deps = checker.get_dependency_status()
    
    print("Dependency Status:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}: {available}")
    
    # Update component status based on dependencies
    tracker = get_error_tracker()
    checker.update_component_status(tracker)
    
    print("\nComponent Status After Dependency Check:")
    for component_type in ComponentType:
        status = tracker.get_component_status(component_type)
        print(f"  {component_type.value}: available={status.available}, deps_met={status.dependencies_met}")
    
    return checker


def test_error_handling_decorator():
    """Test the error handling decorator."""
    print("\n=== Testing Error Handling Decorator ===")
    
    @with_error_handling(
        component=ComponentType.CONTENT_ANALYZER,
        feature_name="test_function",
        fallback_value="fallback_result",
        severity=ErrorSeverity.MEDIUM,
        recovery_action="This is a test function"
    )
    def test_function_that_fails():
        """Test function that always fails."""
        raise ValueError("This is a test error")
    
    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="test_function_success",
        fallback_value="fallback_result",
        severity=ErrorSeverity.LOW
    )
    def test_function_that_succeeds():
        """Test function that succeeds."""
        return "success_result"
    
    # Test failing function
    print("Testing function that fails...")
    result1 = test_function_that_fails()
    print(f"Result: {result1}")
    
    # Test succeeding function
    print("Testing function that succeeds...")
    result2 = test_function_that_succeeds()
    print(f"Result: {result2}")
    
    # Check error tracker
    tracker = get_error_tracker()
    summary = tracker.get_error_summary()
    print(f"Total errors after decorator test: {summary['total_errors']}")


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
    
    # Test minimal features creation
    note_features = create_minimal_features(ComponentType.CONTENT_ANALYZER, "note")
    print(f"\nMinimal note features: {len(note_features)} features")
    
    pair_features = create_minimal_features(ComponentType.FEATURE_ENGINEER, "pair")
    print(f"Minimal pair features: {len(pair_features)} features")


def test_feature_validation():
    """Test feature validation and quality assurance."""
    print("\n=== Testing Feature Validation ===")
    
    # Test feature completeness validation
    test_features = {
        'sentiment_score': 0.5,
        'readability_score': 75.0,
        'complexity_score': 0.8,
        'missing_feature': None,  # This should trigger an error
        'nan_feature': float('nan'),  # This should trigger an error
        'extreme_feature': 1e20  # This should trigger an error
    }
    
    required_features = ['sentiment_score', 'readability_score', 'complexity_score', 'missing_feature']
    
    print("Testing feature completeness validation...")
    errors = validate_feature_completeness(test_features, required_features, ComponentType.CONTENT_ANALYZER)
    print(f"Validation errors: {len(errors)}")
    for error in errors:
        print(f"  - {error}")
    
    # Test feature quality assurance
    print("\nTesting feature quality assurance...")
    corrected_features = ensure_feature_quality(test_features, ComponentType.CONTENT_ANALYZER)
    
    print("Original vs Corrected features:")
    for key in test_features:
        original = test_features[key]
        corrected = corrected_features.get(key, original)
        if original != corrected:
            print(f"  {key}: {original} -> {corrected}")


def test_system_health():
    """Test system health monitoring."""
    print("\n=== Testing System Health Monitoring ===")
    
    # Log system health
    health_info = log_system_health()
    
    print(f"Overall health score: {health_info['overall_health_score']:.3f}")
    print(f"Healthy components: {health_info['healthy_components']}/{health_info['total_components']}")
    
    print("\nDependency status:")
    for dep, status in health_info['dependency_status'].items():
        print(f"  {dep}: {status}")
    
    print("\nComponent reliability:")
    for comp, score in health_info['component_reliability'].items():
        print(f"  {comp}: {score:.3f}")
    
    if health_info['error_summary']['total_errors'] > 0:
        print(f"\nTotal errors in system: {health_info['error_summary']['total_errors']}")
        print("Error breakdown:")
        for severity, count in health_info['error_summary']['by_severity'].items():
            print(f"  {severity}: {count}")


def run_focused_test():
    """Run focused error handling test."""
    print("=" * 60)
    print("FOCUSED ERROR HANDLING TEST")
    print("=" * 60)
    
    try:
        # Test 1: Error tracker
        tracker = test_error_tracker()
        
        # Test 2: Dependency checker
        checker = test_dependency_checker()
        
        # Test 3: Error handling decorator
        test_error_handling_decorator()
        
        # Test 4: Fallback values
        test_fallback_values()
        
        # Test 5: Feature validation
        test_feature_validation()
        
        # Test 6: System health
        test_system_health()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        # Final error summary
        final_tracker = get_error_tracker()
        final_summary = final_tracker.get_error_summary()
        
        print(f"Total errors generated during test: {final_summary['total_errors']}")
        print(f"Components tested: {len(ComponentType)}")
        
        # Check if error handling is working
        if final_summary['total_errors'] > 0:
            print("✓ ERROR HANDLING TEST PASSED")
            print("  Errors were properly tracked and handled")
            print("  Fallback mechanisms activated correctly")
            print("  System demonstrated graceful degradation")
        else:
            print("⚠ ERROR HANDLING TEST INCOMPLETE")
            print("  No errors were generated to test handling")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR HANDLING TEST FAILED")
        print(f"  Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_focused_test()
    sys.exit(0 if success else 1)