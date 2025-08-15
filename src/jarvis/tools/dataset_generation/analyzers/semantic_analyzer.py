"""
Semantic analyzer for text feature extraction.

This module provides comprehensive semantic analysis including sentence transformers,
TF-IDF vectorization, and similarity computations for dataset generation.
"""

import numpy as np
import scipy.sparse
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.logging import setup_logging
from ..error_handling import (
    with_error_handling, ComponentType, ErrorSeverity, FallbackValues,
    get_error_tracker
)

logger = setup_logging(__name__)


class SemanticAnalyzer:
    """Comprehensive semantic analysis for notes and note pairs."""
    
    def __init__(self, vector_encoder: VectorEncoder, 
                 max_tfidf_features: int = 1000,
                 min_df: int = 2,
                 max_df: float = 0.8):
        """Initialize the semantic analyzer.
        
        Args:
            vector_encoder: VectorEncoder service for embeddings
            max_tfidf_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
        """
        self.vector_encoder = vector_encoder
        self.max_tfidf_features = max_tfidf_features
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Alphanumeric tokens starting with letter
        )
        
        self._is_fitted = False
        self._tfidf_matrix = None
        self._document_texts = None
        
    def fit_and_transform(self, documents: List[str]) -> Dict[str, Any]:
        """Fit TF-IDF vectorizer and generate both embeddings and TF-IDF features.
        
        Args:
            documents: List of document texts
            
        Returns:
            Dictionary containing embeddings, TF-IDF matrix, and metadata
        """
        logger.info(f"Fitting semantic analyzer on {len(documents)} documents")
        
        if not documents:
            raise ValueError("Cannot fit on empty document list")
        
        # Clean and prepare documents
        cleaned_docs = [self._clean_text(doc) for doc in documents]
        valid_docs = [doc for doc in cleaned_docs if doc.strip()]
        
        if not valid_docs:
            raise ValueError("No valid documents after cleaning")
        
        logger.info(f"Processing {len(valid_docs)} valid documents")
        
        # Generate sentence transformer embeddings
        try:
            embeddings = self.vector_encoder.encode_documents(valid_docs)
            logger.info(f"Generated embeddings: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            embeddings = np.zeros((len(valid_docs), self.vector_encoder.vector_dim), dtype=np.float32)
        
        # Generate TF-IDF features
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_docs)
            self._is_fitted = True
            self._tfidf_matrix = tfidf_matrix
            self._document_texts = valid_docs
            logger.info(f"Generated TF-IDF matrix: {tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF features: {e}")
            # Create empty sparse matrix as fallback
            tfidf_matrix = scipy.sparse.csr_matrix((len(valid_docs), self.max_tfidf_features))
        
        return {
            'embeddings': embeddings,
            'tfidf_matrix': tfidf_matrix,
            'document_count': len(valid_docs),
            'embedding_dim': embeddings.shape[1] if embeddings.size > 0 else 0,
            'tfidf_features': tfidf_matrix.shape[1],
            'tfidf_vocabulary_size': len(self.tfidf_vectorizer.vocabulary_) if self._is_fitted else 0
        }
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence transformer embeddings for texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Embedding matrix as numpy array
        """
        if not texts:
            return np.empty((0, self.vector_encoder.vector_dim), dtype=np.float32)
        
        cleaned_texts = [self._clean_text(text) for text in texts]
        return self.vector_encoder.encode_documents(cleaned_texts)
    
    def compute_tfidf_features(self, texts: List[str]) -> scipy.sparse.spmatrix:
        """Generate TF-IDF feature matrix for texts.
        
        Args:
            texts: List of texts to vectorize
            
        Returns:
            TF-IDF sparse matrix
        """
        if not self._is_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_and_transform first.")
        
        if not texts:
            return scipy.sparse.csr_matrix((0, self.tfidf_vectorizer.max_features or 1000))
        
        cleaned_texts = [self._clean_text(text) for text in texts]
        return self.tfidf_vectorizer.transform(cleaned_texts)
    
    def compute_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Ensure embeddings are the same shape
        if embedding1.shape != embedding2.shape:
            logger.warning(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")
            return 0.0
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def compute_tfidf_similarity(self, tfidf1: scipy.sparse.spmatrix, tfidf2: scipy.sparse.spmatrix) -> float:
        """Compute cosine similarity between TF-IDF vectors.
        
        Args:
            tfidf1: First TF-IDF vector
            tfidf2: Second TF-IDF vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if tfidf1.nnz == 0 or tfidf2.nnz == 0:
            return 0.0
        
        try:
            # Reshape to 2D if needed
            if tfidf1.ndim == 1:
                tfidf1 = tfidf1.reshape(1, -1)
            if tfidf2.ndim == 1:
                tfidf2 = tfidf2.reshape(1, -1)
            
            similarity = cosine_similarity(tfidf1, tfidf2)[0, 0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to compute TF-IDF similarity: {e}")
            return 0.0
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: Matrix of embeddings (n_docs x embedding_dim)
            
        Returns:
            Similarity matrix (n_docs x n_docs)
        """
        if embeddings.size == 0:
            return np.array([])
        
        return cosine_similarity(embeddings)
    
    def compute_tfidf_similarity_matrix(self, tfidf_matrix: scipy.sparse.spmatrix) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for TF-IDF vectors.
        
        Args:
            tfidf_matrix: TF-IDF sparse matrix (n_docs x n_features)
            
        Returns:
            Similarity matrix (n_docs x n_docs)
        """
        if tfidf_matrix.shape[0] == 0:
            return np.array([])
        
        return cosine_similarity(tfidf_matrix)
    
    def get_top_tfidf_terms(self, document_index: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top TF-IDF terms for a specific document.
        
        Args:
            document_index: Index of the document
            top_k: Number of top terms to return
            
        Returns:
            List of (term, score) tuples
        """
        if not self._is_fitted or self._tfidf_matrix is None:
            return []
        
        if document_index >= self._tfidf_matrix.shape[0]:
            return []
        
        # Get TF-IDF scores for the document
        doc_tfidf = self._tfidf_matrix[document_index].toarray().flatten()
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Get top terms
        top_indices = np.argsort(doc_tfidf)[-top_k:][::-1]
        top_terms = [(feature_names[i], doc_tfidf[i]) for i in top_indices if doc_tfidf[i] > 0]
        
        return top_terms
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the TF-IDF vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        if not self._is_fitted:
            return {'fitted': False}
        
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Calculate some basic stats
        if self._tfidf_matrix is not None:
            avg_doc_length = self._tfidf_matrix.nnz / self._tfidf_matrix.shape[0]
            sparsity = 1.0 - (self._tfidf_matrix.nnz / (self._tfidf_matrix.shape[0] * self._tfidf_matrix.shape[1]))
        else:
            avg_doc_length = 0
            sparsity = 1.0
        
        return {
            'fitted': True,
            'vocabulary_size': vocab_size,
            'max_features': self.max_tfidf_features,
            'document_count': self._tfidf_matrix.shape[0] if self._tfidf_matrix is not None else 0,
            'feature_count': self._tfidf_matrix.shape[1] if self._tfidf_matrix is not None else 0,
            'avg_terms_per_doc': avg_doc_length,
            'sparsity': sparsity,
            'sample_terms': feature_names[:10].tolist() if len(feature_names) > 0 else []
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better feature extraction.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove markdown-style links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"(){}[\]]+', ' ', text)
        
        return text.strip()
    
    def extract_pair_features(self, text1: str, text2: str) -> Dict[str, float]:
        """Extract comprehensive similarity features for a pair of texts with error handling.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity features
        """
        features = {
            'semantic_similarity': FallbackValues.SEMANTIC_SIMILARITY,
            'tfidf_similarity': FallbackValues.TFIDF_SIMILARITY,
            'combined_similarity': FallbackValues.COMBINED_SIMILARITY
        }
        
        # Semantic similarity with error handling
        semantic_sim = self._compute_semantic_similarity_safe(text1, text2)
        features['semantic_similarity'] = semantic_sim
        
        # TF-IDF similarity with error handling
        tfidf_sim = self._compute_tfidf_similarity_safe(text1, text2)
        features['tfidf_similarity'] = tfidf_sim
        
        # Compute combined similarity (normalized weighted average)
        semantic_weight = 0.7
        tfidf_weight = 0.3
        
        # Normalize semantic similarity to [0, 1] range (from [-1, 1])
        normalized_semantic = (features['semantic_similarity'] + 1.0) / 2.0
        
        # Combine normalized similarities
        combined = (
            semantic_weight * normalized_semantic +
            tfidf_weight * features['tfidf_similarity']
        )
        
        # Ensure result is in [0, 1] range
        features['combined_similarity'] = max(0.0, min(1.0, combined))
        
        return features
    
    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="semantic_similarity",
        fallback_value=FallbackValues.SEMANTIC_SIMILARITY,
        severity=ErrorSeverity.MEDIUM,
        recovery_action="Check if sentence-transformers is properly installed"
    )
    def _compute_semantic_similarity_safe(self, text1: str, text2: str) -> float:
        """Safely compute semantic similarity with error handling."""
        embeddings = self.compute_embeddings([text1, text2])
        if embeddings.shape[0] == 2:
            return self.compute_semantic_similarity(embeddings[0], embeddings[1])
        return FallbackValues.SEMANTIC_SIMILARITY
    
    @with_error_handling(
        component=ComponentType.SEMANTIC_ANALYZER,
        feature_name="tfidf_similarity",
        fallback_value=FallbackValues.TFIDF_SIMILARITY,
        severity=ErrorSeverity.LOW,
        recovery_action="TF-IDF similarity will use default value"
    )
    def _compute_tfidf_similarity_safe(self, text1: str, text2: str) -> float:
        """Safely compute TF-IDF similarity with error handling."""
        if not self._is_fitted:
            return FallbackValues.TFIDF_SIMILARITY
        
        tfidf_vectors = self.compute_tfidf_features([text1, text2])
        if tfidf_vectors.shape[0] == 2:
            return self.compute_tfidf_similarity(tfidf_vectors[0], tfidf_vectors[1])
        return FallbackValues.TFIDF_SIMILARITY