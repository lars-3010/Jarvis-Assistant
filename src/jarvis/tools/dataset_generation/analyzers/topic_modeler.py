"""
Topic modeling analyzer for discovering thematic patterns in content.

This module provides topic modeling capabilities using BERTopic or LDA,
with automatic topic assignment and probability extraction for notes.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from jarvis.utils.logging import setup_logging
from ..error_handling import (
    with_error_handling, ComponentType, ErrorSeverity, FallbackValues,
    get_error_tracker, get_dependency_checker
)

logger = setup_logging(__name__)


@dataclass
class TopicModelResult:
    """Results from topic modeling analysis."""
    # Topic assignments
    topic_assignments: List[int] = field(default_factory=list)  # Topic ID for each document
    topic_probabilities: List[List[float]] = field(default_factory=list)  # Probability distribution per document
    
    # Topic information
    topic_count: int = 0
    topic_words: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)  # Top words per topic
    topic_labels: Dict[int, str] = field(default_factory=dict)  # Human-readable topic labels
    
    # Model metadata
    model_type: str = "none"  # "bertopic", "lda", "kmeans", "none"
    confidence_threshold: float = 0.1
    coherence_score: float = 0.0
    
    # Document clustering
    cluster_assignments: List[int] = field(default_factory=list)  # Cluster ID for each document
    cluster_centers: Optional[np.ndarray] = None
    
    def get_dominant_topic(self, doc_index: int) -> Tuple[int, float]:
        """Get the dominant topic for a document.
        
        Args:
            doc_index: Index of the document
            
        Returns:
            Tuple of (topic_id, probability)
        """
        if doc_index >= len(self.topic_probabilities):
            return -1, 0.0
        
        probs = self.topic_probabilities[doc_index]
        if not probs:
            return -1, 0.0
        
        max_idx = np.argmax(probs)
        return max_idx, probs[max_idx]
    
    def get_topic_summary(self, topic_id: int, top_words: int = 5) -> Dict[str, Any]:
        """Get a summary of a specific topic.
        
        Args:
            topic_id: ID of the topic
            top_words: Number of top words to include
            
        Returns:
            Dictionary with topic summary
        """
        if topic_id not in self.topic_words:
            return {"topic_id": topic_id, "label": "Unknown", "words": [], "word_count": 0}
        
        words = self.topic_words[topic_id][:top_words]
        label = self.topic_labels.get(topic_id, f"Topic {topic_id}")
        
        return {
            "topic_id": topic_id,
            "label": label,
            "words": [{"word": word, "score": score} for word, score in words],
            "word_count": len(words)
        }


class TopicModeler:
    """Topic modeling for discovering thematic patterns in document collections."""
    
    def __init__(self, method: str = "auto", n_topics: Optional[int] = None,
                 min_topic_size: int = 5, random_state: int = 42):
        """Initialize the topic modeler.
        
        Args:
            method: Topic modeling method ("bertopic", "lda", "kmeans", "auto")
            n_topics: Number of topics (None for automatic detection)
            min_topic_size: Minimum documents per topic
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.random_state = random_state
        
        # Model instances
        self._bertopic_model = None
        self._lda_model = None
        self._kmeans_model = None
        self._vectorizer = None
        
        # Availability flags
        self._bertopic_available = False
        self._sklearn_available = False
        
        # Check dependencies
        self._check_dependencies()
        
        # Fitted state
        self._is_fitted = False
        self._fitted_method = None
        self._document_embeddings = None
        self._document_texts = None
    
    def _check_dependencies(self):
        """Check availability of topic modeling dependencies."""
        # Check BERTopic
        try:
            import bertopic
            self._bertopic_available = True
            logger.info("BERTopic available for topic modeling")
        except ImportError:
            logger.warning("BERTopic not available. Install with: pip install bertopic")
            self._bertopic_available = False
        
        # Check scikit-learn
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._sklearn_available = True
            logger.info("scikit-learn available for topic modeling")
        except ImportError:
            logger.warning("scikit-learn not available. Install with: pip install scikit-learn")
            self._sklearn_available = False
    
    def fit_topics(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Fit topic model to documents with comprehensive error handling.
        
        Args:
            documents: List of document texts
            embeddings: Optional pre-computed embeddings
            
        Returns:
            TopicModelResult with fitted model information
        """
        if not documents:
            logger.warning("No documents provided for topic modeling")
            return TopicModelResult()
        
        if len(documents) < self.min_topic_size:
            logger.warning(f"Too few documents ({len(documents)}) for topic modeling (min: {self.min_topic_size})")
            return TopicModelResult()
        
        logger.info(f"Fitting topic model on {len(documents)} documents using method: {self.method}")
        
        # Store documents and embeddings
        self._document_texts = documents
        self._document_embeddings = embeddings
        
        # Determine method to use
        method_to_use = self._determine_method()
        
        # Fit the appropriate model with error handling and fallback
        result = self._fit_topic_model_safe(method_to_use, documents, embeddings)
        
        # If BERTopic fails and we're in auto mode, try LDA fallback
        if result.topic_count == 0 and self.method == "auto" and method_to_use == "bertopic" and self._sklearn_available:
            logger.info("BERTopic failed, falling back to LDA for auto method")
            result = self._fit_topic_model_safe("lda", documents, embeddings)
            if result.topic_count > 0:
                method_to_use = "lda"
        
        # If LDA also fails in auto mode, try K-means as final fallback
        if result.topic_count == 0 and self.method == "auto" and self._sklearn_available:
            logger.info("LDA failed, falling back to K-means for auto method")
            result = self._fit_topic_model_safe("kmeans", documents, embeddings)
            if result.topic_count > 0:
                method_to_use = "kmeans"
        
        if result.topic_count > 0:
            self._is_fitted = True
            self._fitted_method = method_to_use
            logger.info(f"Topic modeling complete: {result.topic_count} topics found using {method_to_use}")
        else:
            logger.warning("Topic modeling failed - no topics found")
        
        return result
    
    @with_error_handling(
        component=ComponentType.TOPIC_MODELER,
        feature_name="topic_model_fitting",
        fallback_value=TopicModelResult(),
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Check if BERTopic or scikit-learn dependencies are available"
    )
    def _fit_topic_model_safe(self, method: str, documents: List[str], embeddings: Optional[np.ndarray]) -> TopicModelResult:
        """Safely fit topic model with error handling."""
        if method == "bertopic":
            return self._fit_bertopic(documents, embeddings)
        elif method == "lda":
            return self._fit_lda(documents)
        elif method == "kmeans":
            return self._fit_kmeans(documents, embeddings)
        else:
            logger.warning("No suitable topic modeling method available")
            return TopicModelResult()
    
    def _determine_method(self) -> str:
        """Determine which topic modeling method to use."""
        if self.method == "bertopic" and self._bertopic_available:
            return "bertopic"
        elif self.method == "lda" and self._sklearn_available:
            return "lda"
        elif self.method == "kmeans" and self._sklearn_available:
            return "kmeans"
        elif self.method == "auto":
            # Auto-select best available method
            if self._bertopic_available:
                return "bertopic"
            elif self._sklearn_available:
                return "lda"
            else:
                return "none"
        else:
            return "none"
    
    def _fit_bertopic(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Fit BERTopic model."""
        try:
            from bertopic import BERTopic
            
            # Configure BERTopic with proper settings for small datasets
            try:
                from umap import UMAP
                from hdbscan import HDBSCAN
                
                # Configure UMAP for small datasets
                n_docs = len(documents)
                n_neighbors = min(15, max(2, n_docs - 1))
                n_components = min(5, max(2, n_docs - 1))
                
                umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=self.random_state
                )
                
                # Configure HDBSCAN for small datasets
                min_cluster_size = max(2, min(self.min_topic_size, n_docs // 3))
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Configure BERTopic with custom components
                if embeddings is not None:
                    self._bertopic_model = BERTopic(
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        min_topic_size=self.min_topic_size,
                        calculate_probabilities=True,
                        verbose=False
                    )
                    topics, probabilities = self._bertopic_model.fit_transform(documents, embeddings)
                else:
                    self._bertopic_model = BERTopic(
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        min_topic_size=self.min_topic_size,
                        calculate_probabilities=True,
                        verbose=False
                    )
                    topics, probabilities = self._bertopic_model.fit_transform(documents)
                
            except ImportError:
                # Fallback to default BERTopic configuration if UMAP/HDBSCAN not available
                logger.warning("UMAP or HDBSCAN not available, using default BERTopic configuration")
                
                if embeddings is not None:
                    self._bertopic_model = BERTopic(
                        min_topic_size=self.min_topic_size,
                        calculate_probabilities=True,
                        verbose=False
                    )
                    topics, probabilities = self._bertopic_model.fit_transform(documents, embeddings)
                else:
                    self._bertopic_model = BERTopic(
                        min_topic_size=self.min_topic_size,
                        calculate_probabilities=True,
                        verbose=False
                    )
                    topics, probabilities = self._bertopic_model.fit_transform(documents)
            
            # Extract topic information
            topic_info = self._bertopic_model.get_topic_info()
            topic_words = {}
            topic_labels = {}
            
            for topic_id in topic_info['Topic'].values:
                if topic_id != -1:  # Skip outlier topic
                    words = self._bertopic_model.get_topic(topic_id)
                    topic_words[topic_id] = words[:10]  # Top 10 words
                    
                    # Generate topic label from top words
                    top_words = [word for word, _ in words[:3]]
                    topic_labels[topic_id] = f"Topic {topic_id}: {', '.join(top_words)}"
            
            # Create result
            result = TopicModelResult(
                topic_assignments=topics.tolist() if hasattr(topics, 'tolist') else topics,
                topic_probabilities=probabilities.tolist() if probabilities is not None and hasattr(probabilities, 'tolist') else (probabilities if probabilities is not None else []),
                topic_count=len(topic_words),
                topic_words=topic_words,
                topic_labels=topic_labels,
                model_type="bertopic",
                cluster_assignments=topics.tolist() if hasattr(topics, 'tolist') else topics  # Topics can serve as clusters
            )
            
            return result
            
        except Exception as e:
            logger.error(f"BERTopic fitting failed: {e}")
            return TopicModelResult()
    
    def _fit_lda(self, documents: List[str]) -> TopicModelResult:
        """Fit LDA model using scikit-learn."""
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Determine number of topics
            n_topics = self.n_topics or min(10, max(2, len(documents) // 10))
            
            # Vectorize documents
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = self._vectorizer.fit_transform(documents)
            
            # Fit LDA model
            self._lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=self.random_state,
                max_iter=100,
                learning_method='batch'
            )
            
            self._lda_model.fit(doc_term_matrix)
            
            # Get topic assignments and probabilities
            doc_topic_probs = self._lda_model.transform(doc_term_matrix)
            topic_assignments = np.argmax(doc_topic_probs, axis=1).tolist()
            
            # Extract topic words
            feature_names = self._vectorizer.get_feature_names_out()
            topic_words = {}
            topic_labels = {}
            
            for topic_idx, topic in enumerate(self._lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                topic_words[topic_idx] = top_words
                
                # Generate topic label
                top_word_names = [word for word, _ in top_words[:3]]
                topic_labels[topic_idx] = f"Topic {topic_idx}: {', '.join(top_word_names)}"
            
            # Create result
            result = TopicModelResult(
                topic_assignments=topic_assignments,
                topic_probabilities=doc_topic_probs.tolist(),
                topic_count=n_topics,
                topic_words=topic_words,
                topic_labels=topic_labels,
                model_type="lda",
                cluster_assignments=topic_assignments
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LDA fitting failed: {e}")
            return TopicModelResult()
    
    def _fit_kmeans(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Fit K-means clustering as a simple topic modeling approach."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Determine number of clusters
            n_clusters = self.n_topics or min(8, max(2, len(documents) // 15))
            
            # Use embeddings if available, otherwise use TF-IDF
            if embeddings is not None and embeddings.shape[0] == len(documents):
                features = embeddings
            else:
                # Fallback to TF-IDF
                self._vectorizer = TfidfVectorizer(
                    max_features=500,
                    min_df=2,
                    max_df=0.8,
                    stop_words='english'
                )
                features = self._vectorizer.fit_transform(documents).toarray()
            
            # Fit K-means
            self._kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            
            cluster_assignments = self._kmeans_model.fit_predict(features)
            
            # Generate topic words using TF-IDF if we have the vectorizer
            topic_words = {}
            topic_labels = {}
            
            if self._vectorizer is not None:
                feature_names = self._vectorizer.get_feature_names_out()
                
                for cluster_id in range(n_clusters):
                    # Find documents in this cluster
                    cluster_docs = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
                    
                    if cluster_docs:
                        # Get average TF-IDF scores for this cluster
                        cluster_tfidf = self._vectorizer.transform([documents[i] for i in cluster_docs])
                        avg_scores = np.mean(cluster_tfidf.toarray(), axis=0)
                        
                        # Get top words
                        top_indices = avg_scores.argsort()[-10:][::-1]
                        top_words = [(feature_names[i], avg_scores[i]) for i in top_indices if avg_scores[i] > 0]
                        
                        topic_words[cluster_id] = top_words
                        
                        # Generate label
                        top_word_names = [word for word, _ in top_words[:3]]
                        topic_labels[cluster_id] = f"Cluster {cluster_id}: {', '.join(top_word_names)}"
            else:
                # Simple labels without word analysis
                for cluster_id in range(n_clusters):
                    topic_labels[cluster_id] = f"Cluster {cluster_id}"
            
            # Create pseudo-probabilities (binary assignment)
            topic_probabilities = []
            for assignment in cluster_assignments:
                probs = [0.0] * n_clusters
                probs[assignment] = 1.0
                topic_probabilities.append(probs)
            
            # Create result
            result = TopicModelResult(
                topic_assignments=cluster_assignments.tolist(),
                topic_probabilities=topic_probabilities,
                topic_count=n_clusters,
                topic_words=topic_words,
                topic_labels=topic_labels,
                model_type="kmeans",
                cluster_assignments=cluster_assignments.tolist(),
                cluster_centers=self._kmeans_model.cluster_centers_
            )
            
            return result
            
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return TopicModelResult()
    
    def predict_topics(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Predict topics for new documents using fitted model.
        
        Args:
            documents: List of new document texts
            embeddings: Optional pre-computed embeddings
            
        Returns:
            TopicModelResult with predictions
        """
        if not self._is_fitted:
            logger.warning("Topic model not fitted. Call fit_topics first.")
            return TopicModelResult()
        
        if not documents:
            return TopicModelResult()
        
        logger.info(f"Predicting topics for {len(documents)} documents using {self._fitted_method}")
        
        try:
            if self._fitted_method == "bertopic" and self._bertopic_model is not None:
                return self._predict_bertopic(documents, embeddings)
            elif self._fitted_method == "lda" and self._lda_model is not None:
                return self._predict_lda(documents)
            elif self._fitted_method == "kmeans" and self._kmeans_model is not None:
                return self._predict_kmeans(documents, embeddings)
            else:
                logger.warning("No fitted model available for prediction")
                return TopicModelResult()
                
        except Exception as e:
            logger.error(f"Topic prediction failed: {e}")
            return TopicModelResult()
    
    def _predict_bertopic(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Predict topics using BERTopic model."""
        if embeddings is not None:
            topics, probabilities = self._bertopic_model.transform(documents, embeddings)
        else:
            topics, probabilities = self._bertopic_model.transform(documents)
        
        return TopicModelResult(
            topic_assignments=topics.tolist() if hasattr(topics, 'tolist') else topics,
            topic_probabilities=probabilities.tolist() if probabilities is not None and hasattr(probabilities, 'tolist') else (probabilities if probabilities is not None else []),
            model_type="bertopic",
            cluster_assignments=topics.tolist() if hasattr(topics, 'tolist') else topics
        )
    
    def _predict_lda(self, documents: List[str]) -> TopicModelResult:
        """Predict topics using LDA model."""
        doc_term_matrix = self._vectorizer.transform(documents)
        doc_topic_probs = self._lda_model.transform(doc_term_matrix)
        topic_assignments = np.argmax(doc_topic_probs, axis=1).tolist()
        
        return TopicModelResult(
            topic_assignments=topic_assignments,
            topic_probabilities=doc_topic_probs.tolist(),
            model_type="lda",
            cluster_assignments=topic_assignments
        )
    
    def _predict_kmeans(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> TopicModelResult:
        """Predict clusters using K-means model."""
        if embeddings is not None and embeddings.shape[0] == len(documents):
            features = embeddings
        else:
            features = self._vectorizer.transform(documents).toarray()
        
        cluster_assignments = self._kmeans_model.predict(features)
        
        # Create pseudo-probabilities
        topic_probabilities = []
        for assignment in cluster_assignments:
            probs = [0.0] * self._kmeans_model.n_clusters
            probs[assignment] = 1.0
            topic_probabilities.append(probs)
        
        return TopicModelResult(
            topic_assignments=cluster_assignments.tolist(),
            topic_probabilities=topic_probabilities,
            model_type="kmeans",
            cluster_assignments=cluster_assignments.tolist()
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "fitted": self._is_fitted,
            "method": self._fitted_method,
            "available_methods": {
                "bertopic": self._bertopic_available,
                "lda": self._sklearn_available,
                "kmeans": self._sklearn_available
            },
            "document_count": len(self._document_texts) if self._document_texts else 0,
            "has_embeddings": self._document_embeddings is not None
        }
    
    def export_topics_json(self, result: TopicModelResult) -> str:
        """Export topic modeling results to JSON string.
        
        Args:
            result: TopicModelResult to export
            
        Returns:
            JSON string representation
        """
        export_data = {
            "model_type": result.model_type,
            "topic_count": result.topic_count,
            "topics": {}
        }
        
        for topic_id, words in result.topic_words.items():
            export_data["topics"][str(topic_id)] = {
                "label": result.topic_labels.get(topic_id, f"Topic {topic_id}"),
                "words": [{"word": word, "score": float(score)} for word, score in words[:5]]
            }
        
        return json.dumps(export_data, indent=2)