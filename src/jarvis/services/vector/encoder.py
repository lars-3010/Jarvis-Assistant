"""
Vector encoding service using sentence transformers.

This module provides text encoding capabilities using pre-trained sentence transformer
models, with support for chunking, device detection, and error handling.
"""

import re
import uuid
from collections.abc import Sequence
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from jarvis.utils.logging import setup_logging
from jarvis.utils.config import get_settings
from jarvis.utils.errors import JarvisError, ServiceError
from jarvis.core.interfaces import IVectorEncoder

logger = setup_logging(__name__)


class VectorEncoder(IVectorEncoder):
    """Enhanced encoder with device detection, chunking, and error handling."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the vector encoder.
        
        Args:
            model_name: Sentence transformer model name
            device: PyTorch device (auto-detected if None)
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or self._detect_device()
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.vector_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized encoder with model: {self.model_name} on device: {self.device}")
            logger.info(f"Embedding dimension: {self.vector_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise ServiceError(f"Failed to initialize encoder with model {self.model_name}: {e}") from e
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        settings = get_settings()
        
        # Check settings first
        if hasattr(settings, 'embedding_device') and settings.embedding_device:
            device = settings.embedding_device.lower()
            if device in ['cpu', 'cuda', 'mps']:
                # Validate device availability
                if device == 'cuda' and torch.cuda.is_available():
                    return 'cuda'
                elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'
                elif device == 'cpu':
                    return 'cpu'
        
        # Auto-detection fallback
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into a vector with error handling.
        
        Args:
            query: Query text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if not query or not query.strip():
            logger.warning("Attempted to encode empty query")
            return np.zeros(self.vector_dim, dtype=np.float32)
        
        # Clean and truncate query
        cleaned_query = self._clean_text(query)
        truncated_query = self._truncate_text(cleaned_query, max_length=8000)
        
        try:
            embedding = self.model.encode(truncated_query, show_progress_bar=False)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding query '{query[:50]}...': {e}")
            raise ServiceError(f"Error encoding query: {e}") from e

    def encode_documents(
        self, 
        documents: Sequence[str], 
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode a sequence of documents into a matrix with error handling.
        
        Args:
            documents: Sequence of document texts
            batch_size: Batch size for encoding (from settings if None)
            
        Returns:
            Matrix of embeddings as numpy array
        """
        if not documents:
            return np.empty((0, self.vector_dim), dtype=np.float32)
        
        # Get batch size from settings if not provided
        if batch_size is None:
            settings = get_settings()
            batch_size = getattr(settings, 'embedding_batch_size', 32)
        
        # Filter and clean documents
        valid_docs = []
        for doc in documents:
            if doc and doc.strip():
                cleaned = self._clean_text(doc)
                truncated = self._truncate_text(cleaned, max_length=8000)
                valid_docs.append(truncated)
            else:
                valid_docs.append("")  # Placeholder for empty docs
        
        if not valid_docs:
            return np.empty((0, self.vector_dim), dtype=np.float32)
        
        try:
            embeddings = self.model.encode(
                valid_docs, 
                show_progress_bar=False, 
                batch_size=batch_size
            )
            logger.debug(f"Encoded {len(valid_docs)} documents in batches of {batch_size}")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding {len(valid_docs)} documents: {e}")
            raise ServiceError(f"Error encoding documents: {e}") from e
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200,
        min_chunk_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for better embedding.
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to include
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        if not text or len(text) < min_chunk_size:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine chunk end
            end = min(start + chunk_size, len(text))
            
            # Try to find a natural breakpoint if not at the end of text
            if end < len(text):
                # Search for sentence endings in the latter half of the chunk
                search_start = max(start + chunk_size // 2, start)
                search_text = text[search_start:end]
                
                # Look for sentence endings in order of preference
                breakpoints = ['. ', '.\n', '? ', '! ', '\n\n', '\n', '. ']
                found_break = False
                
                for pattern in breakpoints:
                    pos = search_text.rfind(pattern)
                    if pos >= 0:  # Found a breakpoint
                        end = search_start + pos + len(pattern)
                        found_break = True
                        break
                
                # If no good breakpoint found, try word boundaries
                if not found_break:
                    word_boundary = search_text.rfind(' ')
                    if word_boundary >= 0:
                        end = search_start + word_boundary
            
            # Extract the chunk
            chunk_text = text[start:end].strip()
            
            # Only create chunks with meaningful content
            if chunk_text and len(chunk_text) >= min_chunk_size:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "length": len(chunk_text),
                    "chunk_index": chunk_index
                })
                chunk_index += 1
            
            # Move to next chunk position, with overlap
            start = max(end - overlap, start + 1)  # Ensure progress
        
        logger.debug(f"Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def encode_chunks(
        self, 
        text: str,
        chunk_size: int = 1000, 
        overlap: int = 200,
        min_chunk_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Create embeddings for chunks of text.
        
        Args:
            text: Text to process
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to include
            
        Returns:
            List of dictionaries with chunk text, metadata and embeddings
        """
        # Get chunks
        chunks = self.chunk_text(text, chunk_size, overlap, min_chunk_size)
        
        if not chunks:
            return []
        
        # Extract text for batch encoding
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Create embeddings for all chunks
        embeddings = self.encode_documents(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk["embedding"] = embeddings[i].tolist()
            else:
                logger.warning(f"Missing embedding for chunk {i}")
                chunk["embedding"] = np.zeros(self.vector_dim, dtype=np.float32).tolist()
        
        return chunks
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Ensure embeddings are the same shape
        if embedding1.shape != embedding2.shape:
            logger.error(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")
            return 0.0
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def _truncate_text(self, text: str, max_length: int = 8000) -> str:
        """Truncate text if it exceeds maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at a sentence boundary
        truncated = text[:max_length]
        last_sentence = truncated.rfind('. ')
        if last_sentence > max_length * 0.8:  # Only if we don't lose too much
            return truncated[:last_sentence + 1]
        
        return truncated
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better embedding quality.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (but keep domain names that might be important)
        text = re.sub(r'https?://\S+', '[URL]', text)
        
        # Remove markdown-style links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove excessive special characters
        text = re.sub(r'[^\w\s.,!?;:\-\'\"(){}[\]]+', '', text)
        
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'vector_dimension': self.vector_dim,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    # Interface methods required by IVectorEncoder
    def encode(self, text: str) -> torch.Tensor:
        """Encode text into a vector representation.
        
        Args:
            text: Text to encode
            
        Returns:
            Torch tensor with the embedding
        """
        embedding = self.encode_query(text)
        return torch.from_numpy(embedding)
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts into vector representations.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Torch tensor with the embeddings
        """
        embeddings = self.encode_documents(texts)
        return torch.from_numpy(embeddings)