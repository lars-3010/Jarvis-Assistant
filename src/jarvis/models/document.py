"""
Document models for Jarvis Assistant.

This module defines data models for documents, notes, and content representation
used across the vector and graph search systems.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    
    file_path: Path
    vault_name: str
    last_modified: datetime
    created: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    
    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat()
        }


class DocumentContent(BaseModel):
    """Content representation of a document."""
    
    title: Optional[str] = None
    content: str
    frontmatter: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)


class DocumentEmbedding(BaseModel):
    """Embedding representation of a document."""
    
    embedding: List[float]
    model_name: str
    chunk_index: int = 0
    chunk_text: Optional[str] = None


class Document(BaseModel):
    """Complete document representation."""
    
    metadata: DocumentMetadata
    content: DocumentContent
    embedding: Optional[DocumentEmbedding] = None
    
    @property
    def relative_path(self) -> Path:
        """Get the relative path within the vault."""
        # Implementation will be added in Phase 2.2
        return self.metadata.file_path
    
    @property
    def title(self) -> str:
        """Get document title, falling back to filename."""
        if self.content.title:
            return self.content.title
        return self.metadata.file_path.stem
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.dict()
    
    @classmethod
    def from_file(cls, file_path: Path, vault_name: str) -> "Document":
        """Create document from file path.
        
        Args:
            file_path: Path to the document file
            vault_name: Name of the vault containing the document
            
        Returns:
            Document instance
        """
        # Implementation will be added in Phase 2.2
        metadata = DocumentMetadata(
            file_path=file_path,
            vault_name=vault_name,
            last_modified=datetime.now(),
            size_bytes=0
        )
        
        content = DocumentContent(
            content="Content loading will be implemented in Phase 2.2"
        )
        
        return cls(metadata=metadata, content=content)


class SearchResult(BaseModel):
    """Represents a search result with metadata."""
    
    vault_name: str
    path: Path
    similarity_score: float
    full_path: Optional[Path] = None
    
    def __init__(self, vault_name: str, path: Path, similarity_score: float, full_path: Optional[Path] = None, **kwargs):
        """Initialize search result.
        
        Args:
            vault_name: Name of the vault
            path: Path relative to vault
            similarity_score: Similarity score from search
            full_path: Full absolute path (optional)
        """
        super().__init__(
            vault_name=vault_name,
            path=path,
            similarity_score=similarity_score,
            full_path=full_path or path,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'vault_name': self.vault_name,
            'path': str(self.path),
            'full_path': str(self.full_path),
            'similarity_score': self.similarity_score
        }
    
    def __repr__(self) -> str:
        return f"SearchResult(vault={self.vault_name}, path={self.path}, score={self.similarity_score:.3f})"
    
    class Config:
        json_encoders = {
            Path: str
        }