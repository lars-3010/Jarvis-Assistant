"""
Comprehensive error handling and fallback mechanisms for dataset generation.

This module provides robust error handling, graceful degradation, and fallback
values for all feature extraction components in the enhanced dataset generation system.
"""

import traceback
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import time

from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import JarvisError

logger = setup_logging(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for feature extraction failures."""
    LOW = "low"           # Non-critical feature, has good fallback
    MEDIUM = "medium"     # Important feature, degraded fallback
    HIGH = "high"         # Critical feature, minimal fallback
    CRITICAL = "critical" # Essential feature, system may fail


class ComponentType(Enum):
    """Types of components that can fail."""
    SEMANTIC_ANALYZER = "semantic_analyzer"
    CONTENT_ANALYZER = "content_analyzer"
    TOPIC_MODELER = "topic_modeler"
    GRAPH_ANALYZER = "graph_analyzer"
    FEATURE_ENGINEER = "feature_engineer"
    DEPENDENCY = "dependency"


@dataclass
class FeatureExtractionError:
    """Detailed information about a feature extraction error."""
    component: ComponentType
    feature_name: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    fallback_used: bool = False
    fallback_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'component': self.component.value,
            'feature_name': self.feature_name,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'timestamp': self.timestamp,
            'context': self.context,
            'recovery_action': self.recovery_action,
            'fallback_used': self.fallback_used,
            'fallback_value': str(self.fallback_value) if self.fallback_value is not None else None
        }


@dataclass
class ComponentStatus:
    """Status of a component and its capabilities."""
    component: ComponentType
    available: bool
    dependencies_met: bool
    last_error: Optional[FeatureExtractionError] = None
    error_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this component."""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability score (0-1) based on performance."""
        if not self.available or not self.dependencies_met:
            return 0.0
        
        success_rate = self.success_rate
        fallback_penalty = min(0.5, self.fallback_count * 0.1)
        return max(0.0, success_rate - fallback_penalty)


class ErrorTracker:
    """Tracks errors and component status across the system."""
    
    def __init__(self):
        self.errors: List[FeatureExtractionError] = []
        self.component_status: Dict[ComponentType, ComponentStatus] = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize component status tracking."""
        for component in ComponentType:
            self.component_status[component] = ComponentStatus(
                component=component,
                available=True,
                dependencies_met=True
            )
    
    def record_error(self, error: FeatureExtractionError):
        """Record an error and update component status."""
        self.errors.append(error)
        
        # Update component status
        if error.component in self.component_status:
            status = self.component_status[error.component]
            status.last_error = error
            status.error_count += 1
            
            if error.fallback_used:
                status.fallback_count += 1
            
            # Mark component as unavailable if too many critical errors
            if error.severity == ErrorSeverity.CRITICAL:
                status.available = False
                logger.critical(f"Component {error.component.value} marked as unavailable due to critical error")
    
    def record_success(self, component: ComponentType):
        """Record a successful operation."""
        if component in self.component_status:
            self.component_status[component].success_count += 1
    
    def get_component_status(self, component: ComponentType) -> ComponentStatus:
        """Get status for a specific component."""
        return self.component_status.get(component, ComponentStatus(
            component=component,
            available=False,
            dependencies_met=False
        ))
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.errors:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_component': {},
                'recent_errors': []
            }
        
        # Count by severity
        by_severity = {}
        for error in self.errors:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by component
        by_component = {}
        for error in self.errors:
            component = error.component.value
            by_component[component] = by_component.get(component, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = [error.to_dict() for error in self.errors[-10:]]
        
        return {
            'total_errors': len(self.errors),
            'by_severity': by_severity,
            'by_component': by_component,
            'recent_errors': recent_errors,
            'component_reliability': {
                comp.value: status.reliability_score 
                for comp, status in self.component_status.items()
            }
        }


# Global error tracker instance
_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _error_tracker


def with_error_handling(
    component: ComponentType,
    feature_name: str,
    fallback_value: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    recovery_action: Optional[str] = None
):
    """Decorator for robust error handling with fallbacks.
    
    Args:
        component: Component type that might fail
        feature_name: Name of the feature being extracted
        fallback_value: Value to return if extraction fails
        severity: Severity level of potential failures
        recovery_action: Suggested recovery action for users
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                _error_tracker.record_success(component)
                
                # Log successful extraction for debugging
                elapsed = time.time() - start_time
                logger.debug(f"Successfully extracted {feature_name} in {elapsed:.3f}s")
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                
                # Create detailed error record
                error = FeatureExtractionError(
                    component=component,
                    feature_name=feature_name,
                    error_type=type(e).__name__,
                    message=str(e),
                    severity=severity,
                    timestamp=time.time(),
                    context={
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'elapsed_seconds': elapsed
                    },
                    stack_trace=traceback.format_exc(),
                    recovery_action=recovery_action,
                    fallback_used=fallback_value is not None,
                    fallback_value=fallback_value
                )
                
                # Record the error
                _error_tracker.record_error(error)
                
                # Log based on severity
                if severity == ErrorSeverity.CRITICAL:
                    logger.critical(f"Critical failure in {feature_name}: {e}")
                elif severity == ErrorSeverity.HIGH:
                    logger.error(f"High severity failure in {feature_name}: {e}")
                elif severity == ErrorSeverity.MEDIUM:
                    logger.warning(f"Medium severity failure in {feature_name}: {e}")
                else:
                    logger.debug(f"Low severity failure in {feature_name}: {e}")
                
                # Return fallback value or re-raise
                if fallback_value is not None:
                    logger.info(f"Using fallback value for {feature_name}: {fallback_value}")
                    return fallback_value
                else:
                    raise
        
        return wrapper
    return decorator


class FallbackValues:
    """Centralized fallback values for different feature types."""
    
    # Semantic features
    SEMANTIC_SIMILARITY = 0.0
    TFIDF_SIMILARITY = 0.0
    COMBINED_SIMILARITY = 0.0
    EMBEDDING_DIMENSION = 384  # Common sentence transformer dimension
    
    # Content features
    SENTIMENT_SCORE = 0.0
    SENTIMENT_LABEL = "neutral"
    READABILITY_SCORE = 50.0  # Average readability
    READABILITY_GRADE = 8.0   # 8th grade level
    COMPLEXITY_SCORE = 0.5    # Medium complexity
    VOCABULARY_RICHNESS = 0.5 # Average richness
    CONTENT_TYPE = "general"
    TECHNICAL_DENSITY = 0.0
    CONCEPT_DENSITY = 0.0
    
    # Topic modeling features
    DOMINANT_TOPIC_ID = -1
    DOMINANT_TOPIC_PROBABILITY = 0.0
    TOPIC_LABEL = "Unknown Topic"
    TOPIC_PROBABILITIES = []
    TOPIC_SIMILARITY = 0.0
    
    # Graph features
    PAGERANK = 1.0 / 1000  # Assume 1000 nodes for default
    BETWEENNESS_CENTRALITY = 0.0
    CLOSENESS_CENTRALITY = 0.0
    CLUSTERING_COEFFICIENT = 0.0
    EIGENVECTOR_CENTRALITY = 0.0
    COMMUNITY_ID = -1
    BRIDGE_SCORE = 0.0
    
    # TF-IDF features
    TFIDF_VOCABULARY_RICHNESS = 0.0
    AVG_TFIDF_SCORE = 0.0
    TOP_TFIDF_TERMS = []
    
    # Named entities
    NAMED_ENTITIES = []
    ENTITY_TYPES = {}
    
    # Structure features
    HEADING_COUNT = 0
    MAX_HEADING_DEPTH = 0
    LIST_COUNT = 0
    CODE_BLOCK_COUNT = 0
    
    @classmethod
    def get_fallback_content_features(cls) -> Dict[str, Any]:
        """Get fallback values for content analysis features."""
        return {
            'sentiment_score': cls.SENTIMENT_SCORE,
            'sentiment_label': cls.SENTIMENT_LABEL,
            'readability_score': cls.READABILITY_SCORE,
            'readability_grade': cls.READABILITY_GRADE,
            'complexity_score': cls.COMPLEXITY_SCORE,
            'vocabulary_richness': cls.VOCABULARY_RICHNESS,
            'content_type': cls.CONTENT_TYPE,
            'technical_density': cls.TECHNICAL_DENSITY,
            'concept_density': cls.CONCEPT_DENSITY,
            'named_entities': cls.NAMED_ENTITIES,
            'entity_types': cls.ENTITY_TYPES,
            'heading_count': cls.HEADING_COUNT,
            'max_heading_depth': cls.MAX_HEADING_DEPTH
        }
    
    @classmethod
    def get_fallback_topic_features(cls) -> Dict[str, Any]:
        """Get fallback values for topic modeling features."""
        return {
            'dominant_topic_id': cls.DOMINANT_TOPIC_ID,
            'dominant_topic_probability': cls.DOMINANT_TOPIC_PROBABILITY,
            'topic_label': cls.TOPIC_LABEL,
            'topic_probabilities': cls.TOPIC_PROBABILITIES,
            'topic_similarity': cls.TOPIC_SIMILARITY
        }
    
    @classmethod
    def get_fallback_graph_features(cls) -> Dict[str, Any]:
        """Get fallback values for graph analysis features."""
        return {
            'pagerank': cls.PAGERANK,
            'betweenness_centrality': cls.BETWEENNESS_CENTRALITY,
            'closeness_centrality': cls.CLOSENESS_CENTRALITY,
            'clustering_coefficient': cls.CLUSTERING_COEFFICIENT,
            'eigenvector_centrality': cls.EIGENVECTOR_CENTRALITY,
            'community_id': cls.COMMUNITY_ID,
            'bridge_score': cls.BRIDGE_SCORE
        }


class DependencyChecker:
    """Checks availability of optional dependencies."""
    
    def __init__(self):
        self._dependency_cache: Dict[str, bool] = {}
    
    def check_spacy(self) -> bool:
        """Check if spaCy and English model are available."""
        if 'spacy' in self._dependency_cache:
            return self._dependency_cache['spacy']
        
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            self._dependency_cache['spacy'] = True
            logger.info("spaCy with English model is available")
            return True
        except (ImportError, OSError) as e:
            self._dependency_cache['spacy'] = False
            logger.warning(f"spaCy not available: {e}")
            return False
    
    def check_textstat(self) -> bool:
        """Check if textstat is available."""
        if 'textstat' in self._dependency_cache:
            return self._dependency_cache['textstat']
        
        try:
            import textstat
            self._dependency_cache['textstat'] = True
            logger.info("textstat is available")
            return True
        except ImportError as e:
            self._dependency_cache['textstat'] = False
            logger.warning(f"textstat not available: {e}")
            return False
    
    def check_bertopic(self) -> bool:
        """Check if BERTopic is available."""
        if 'bertopic' in self._dependency_cache:
            return self._dependency_cache['bertopic']
        
        try:
            import bertopic
            self._dependency_cache['bertopic'] = True
            logger.info("BERTopic is available")
            return True
        except ImportError as e:
            self._dependency_cache['bertopic'] = False
            logger.warning(f"BERTopic not available: {e}")
            return False
    
    def check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        if 'sklearn' in self._dependency_cache:
            return self._dependency_cache['sklearn']
        
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.cluster import KMeans
            self._dependency_cache['sklearn'] = True
            logger.info("scikit-learn is available")
            return True
        except ImportError as e:
            self._dependency_cache['sklearn'] = False
            logger.warning(f"scikit-learn not available: {e}")
            return False
    
    def check_networkx_advanced(self) -> bool:
        """Check if NetworkX with advanced algorithms is available."""
        if 'networkx_advanced' in self._dependency_cache:
            return self._dependency_cache['networkx_advanced']
        
        try:
            import networkx as nx
            # Test advanced algorithms
            G = nx.Graph()
            G.add_edges_from([(1, 2), (2, 3)])
            nx.pagerank(G)
            nx.betweenness_centrality(G)
            self._dependency_cache['networkx_advanced'] = True
            logger.info("NetworkX with advanced algorithms is available")
            return True
        except (ImportError, Exception) as e:
            self._dependency_cache['networkx_advanced'] = False
            logger.warning(f"NetworkX advanced algorithms not available: {e}")
            return False
    
    def get_dependency_status(self) -> Dict[str, bool]:
        """Get status of all dependencies."""
        return {
            'spacy': self.check_spacy(),
            'textstat': self.check_textstat(),
            'bertopic': self.check_bertopic(),
            'sklearn': self.check_sklearn(),
            'networkx_advanced': self.check_networkx_advanced()
        }
    
    def update_component_status(self, error_tracker: ErrorTracker):
        """Update component status based on dependency availability."""
        deps = self.get_dependency_status()
        
        # Update content analyzer status
        content_status = error_tracker.get_component_status(ComponentType.CONTENT_ANALYZER)
        content_status.dependencies_met = deps['spacy'] or deps['textstat']
        content_status.available = content_status.dependencies_met
        
        # Update topic modeler status
        topic_status = error_tracker.get_component_status(ComponentType.TOPIC_MODELER)
        topic_status.dependencies_met = deps['bertopic'] or deps['sklearn']
        topic_status.available = topic_status.dependencies_met
        
        # Update graph analyzer status
        graph_status = error_tracker.get_component_status(ComponentType.GRAPH_ANALYZER)
        graph_status.dependencies_met = deps['networkx_advanced']
        graph_status.available = graph_status.dependencies_met


# Global dependency checker instance
_dependency_checker = DependencyChecker()


def get_dependency_checker() -> DependencyChecker:
    """Get the global dependency checker instance."""
    return _dependency_checker


def validate_feature_completeness(features: Dict[str, Any], 
                                required_features: List[str],
                                component: ComponentType) -> List[str]:
    """Validate that required features are present and valid.
    
    Args:
        features: Dictionary of extracted features
        required_features: List of required feature names
        component: Component that extracted the features
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for feature_name in required_features:
        if feature_name not in features:
            errors.append(f"Missing required feature: {feature_name}")
            continue
        
        value = features[feature_name]
        
        # Check for None values
        if value is None:
            errors.append(f"Feature {feature_name} is None")
            continue
        
        # Type-specific validation
        if isinstance(value, float):
            if not (-1e10 <= value <= 1e10):  # Reasonable range check
                errors.append(f"Feature {feature_name} has extreme value: {value}")
            elif value != value:  # NaN check
                errors.append(f"Feature {feature_name} is NaN")
        
        elif isinstance(value, list) and len(value) == 0 and feature_name.endswith('_list'):
            # Empty lists might be valid for some features
            logger.debug(f"Feature {feature_name} is empty list (may be valid)")
    
    if errors:
        error = FeatureExtractionError(
            component=component,
            feature_name="validation",
            error_type="ValidationError",
            message=f"Feature validation failed: {'; '.join(errors)}",
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time(),
            context={'failed_features': errors}
        )
        _error_tracker.record_error(error)
    
    return errors


def ensure_feature_quality(features: Dict[str, Any], 
                          component: ComponentType) -> Dict[str, Any]:
    """Ensure feature quality by applying corrections and fallbacks.
    
    Args:
        features: Dictionary of features to validate and correct
        component: Component that extracted the features
        
    Returns:
        Corrected features dictionary
    """
    corrected = features.copy()
    corrections_made = []
    
    # Fix common issues
    for key, value in corrected.items():
        if isinstance(value, float):
            # Fix NaN values
            if value != value:  # NaN check
                if 'similarity' in key.lower():
                    corrected[key] = 0.0
                elif 'score' in key.lower():
                    corrected[key] = 0.0
                elif 'probability' in key.lower():
                    corrected[key] = 0.0
                else:
                    corrected[key] = 0.0
                corrections_made.append(f"Fixed NaN in {key}")
            
            # Clamp similarity values to [0, 1] or [-1, 1] range
            elif 'similarity' in key.lower():
                if key == 'semantic_similarity':
                    # Semantic similarity can be [-1, 1]
                    if value < -1.0:
                        corrected[key] = -1.0
                        corrections_made.append(f"Clamped {key} to -1.0")
                    elif value > 1.0:
                        corrected[key] = 1.0
                        corrections_made.append(f"Clamped {key} to 1.0")
                else:
                    # Other similarities should be [0, 1]
                    if value < 0.0:
                        corrected[key] = 0.0
                        corrections_made.append(f"Clamped {key} to 0.0")
                    elif value > 1.0:
                        corrected[key] = 1.0
                        corrections_made.append(f"Clamped {key} to 1.0")
            
            # Clamp probability values to [0, 1]
            elif 'probability' in key.lower() or 'score' in key.lower():
                if value < 0.0:
                    corrected[key] = 0.0
                    corrections_made.append(f"Clamped {key} to 0.0")
                elif value > 1.0 and 'readability' not in key.lower():
                    # Readability scores can be > 1.0
                    corrected[key] = 1.0
                    corrections_made.append(f"Clamped {key} to 1.0")
    
    # Log corrections if any were made
    if corrections_made:
        logger.info(f"Applied {len(corrections_made)} feature corrections for {component.value}")
        logger.debug(f"Corrections: {corrections_made}")
    
    return corrected


def create_minimal_features(component: ComponentType, 
                           feature_type: str = "note") -> Dict[str, Any]:
    """Create minimal fallback features when extraction completely fails.
    
    Args:
        component: Component that failed
        feature_type: Type of features to create ("note" or "pair")
        
    Returns:
        Dictionary with minimal fallback features
    """
    if feature_type == "note":
        if component == ComponentType.CONTENT_ANALYZER:
            return FallbackValues.get_fallback_content_features()
        elif component == ComponentType.TOPIC_MODELER:
            return FallbackValues.get_fallback_topic_features()
        elif component == ComponentType.GRAPH_ANALYZER:
            return FallbackValues.get_fallback_graph_features()
        else:
            return {
                'sentiment_score': FallbackValues.SENTIMENT_SCORE,
                'complexity_score': FallbackValues.COMPLEXITY_SCORE,
                'dominant_topic_id': FallbackValues.DOMINANT_TOPIC_ID,
                'pagerank': FallbackValues.PAGERANK
            }
    
    elif feature_type == "pair":
        return {
            'semantic_similarity': FallbackValues.SEMANTIC_SIMILARITY,
            'tfidf_similarity': FallbackValues.TFIDF_SIMILARITY,
            'combined_similarity': FallbackValues.COMBINED_SIMILARITY,
            'topic_similarity': FallbackValues.TOPIC_SIMILARITY,
            'same_dominant_topic': False,
            'content_similarity': 0.0
        }
    
    return {}


def log_system_health() -> Dict[str, Any]:
    """Log comprehensive system health information.
    
    Returns:
        Dictionary with system health metrics
    """
    error_tracker = get_error_tracker()
    dependency_checker = get_dependency_checker()
    
    # Update component status based on dependencies
    dependency_checker.update_component_status(error_tracker)
    
    # Get health metrics
    error_summary = error_tracker.get_error_summary()
    dependency_status = dependency_checker.get_dependency_status()
    
    # Calculate overall health score
    total_components = len(ComponentType)
    healthy_components = sum(
        1 for status in error_tracker.component_status.values()
        if status.available and status.reliability_score > 0.7
    )
    health_score = healthy_components / total_components if total_components > 0 else 0.0
    
    health_info = {
        'overall_health_score': health_score,
        'healthy_components': healthy_components,
        'total_components': total_components,
        'dependency_status': dependency_status,
        'error_summary': error_summary,
        'component_reliability': {
            comp.value: status.reliability_score
            for comp, status in error_tracker.component_status.items()
        }
    }
    
    # Log health summary
    if health_score >= 0.8:
        logger.info(f"System health: GOOD ({health_score:.2f}) - {healthy_components}/{total_components} components healthy")
    elif health_score >= 0.5:
        logger.warning(f"System health: DEGRADED ({health_score:.2f}) - {healthy_components}/{total_components} components healthy")
    else:
        logger.error(f"System health: POOR ({health_score:.2f}) - {healthy_components}/{total_components} components healthy")
    
    return health_info